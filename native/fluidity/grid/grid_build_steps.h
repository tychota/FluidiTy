// The formulas of a grid build that the CPU builder and the GPU builder share: sort keys, the checks of one sorted
// cell, the split cells that one active cell adds, and the table entries of one block, one cell place and one
// face place.
//
// Purpose: write once each rule a grid build applies to one item, so the CPU builder and every GPU thread run the
//   same code. The CPU and the GPU builders call these functions and nothing else for a table entry; that is why their
//   tables are identical.
//
// Idea: a build sorts the active cells into the fixed order, checks them, adds the split cells and numbers the cell
//   blocks. On the GPU each of those steps is one thread per item, so each rule here looks at one item of a sorted
//   list and its neighbours in the list, and returns a value; the loop around it decides where the value goes.
//   Sort keys. The stable order sorts by unsigned integers, three passes, the least significant first. For a
//   coordinate c of a cell of size level l the biased coordinate is
//     biased(c, l) = (c + 2^(18 + l)) << (15 - l) = c 2^(15 - l) + 2^33,
//   the low corner in widths of a size-level-15 cell, moved up by 2^33. A cell's key is (biased(x, l) << 4) | l in
//   pass 0, biased(y, l) in pass 1 and biased(z, l) in pass 2. After the three passes the cells are ordered by z, then
//   y, then x, and at one low corner the smaller l, the bigger cell, first: the order of ComesBefore. A block's key
//   is the same with the block's low corner, 4 floor(c / 4), in place of c, so every cell of a block has the key of
//   its block.
//   Why the keys are in range, and the same on every machine. A valid c is at least -(2^18 - 1), so
//   c + 2^(18 + l) >= 1. A block's low corner is at least -2^18, so its sum is at least 0, and 0 only for the lowest
//   block of size level 0. Nothing negative is ever shifted. The largest value is (2^18 - 1 + 2^18) << 15 =
//   2^34 - 2^15, so biased < 2^34 and pass 0 is below 2^38. Only integer adds and shifts are used. An id with bit 63
//   clear, such as CellId{0}, has the key 2^64 - 1 in every pass, so it sorts last. Any other id keeps an ordinary
//   key, also when CheckCellId fails it for stray bits 61..62, a size level the frame does not use or the coordinate
//   kMaxCoord + 1; step 1 fails such a list whatever its key, and the sorted order is then never used. The largest
//   stored coordinate, kMaxCoord + 1 = 2^18, gives biased(2^18, 0) = 2^34, so pass 0 is then at most 2^38 + 15, still
//   below 2^39: the key keeps its meaning and nothing overflows.
//   Places beside a cell. A rule that looks at a place beside a cell works on the raw coordinate. It finds the place
//   one size level bigger by floor(c / 2) and a small cell by 2 c or 2 c + 1, and makes an id only for a coordinate
//   in range at its own size level (ComputeIsCoordInRange). Outside that range a packed id would name another cell.
//   A coordinate outside the range holds no cell of its size level. At the low limit nothing lies beyond a cell: the
//   bigger cell that would hold the place beyond holds the cell itself. At the high limit a bigger cell may lie beyond,
//   found at its own size level, where the halved coordinate is in range.
//   Each check of a pair of cells looks only from the smaller cell, so a broken pair is reported at one position.
//   Example: the size-level-0 cell {0,0,0} and the size-level-2 cell {4,4,4} touch at a corner. Sorted, the first is
//   at position 0 and the second at position 1. CheckSizeStepAt at position 1 reports direction 0, (dz, dy, dx) =
//   (-1, -1, -1), with the failure key ToFailureKey(4, 1, 0) = (4 << 56) | (1 << 8).
//   Table entries. Once the block ids are sorted, each block finds its neighbours by binary search, each cell place
//   that holds no active or split cell asks the place that covers it in the bigger block, and each face place looks
//   at the two cell places beside it. A place beside a block is a local coordinate -1 or 4: the same-size-level
//   block on that side holds it, or, when there is none, only a cell one size level bigger can cover it.
//   Example: size level 0 {0,0,0} active beside the 8 size-level-1 cells of {1,0,0}: the size-level-1 places
//   x = 0..1 are kInsideBiggerCell, and the X faces at x = 2 between them and the small cells are 4 kBetweenCells.
//
// Words:
//   sort key          the unsigned integer an item is sorted by in one pass of the stable order
//   eighth-0 chain    from a cell, its eighth-0 small cell, that cell's eighth-0 small cell, and so on: the cells that
//                     share its low corner, one per smaller size level
//   present           of a small cell in a sibling check: active, or split with an active cell at the end of its
//                     eighth-0 chain
//   walk step         one size level of the sibling check's climb from an active cell through the split cells it adds
//   failure key       the number (step, position, detail) whose smallest value names the first failure of a build
//   local coordinate  the coordinate of a cell place in its block, 0..3 on each axis; -1 and 4 are one cell beyond
//   cover             of a place, the place one size level bigger that holds it: floor(c / 2) at size level l - 1
// Owns:    kSortKeyPassCount, ToCellSortKey, ToBlockSortKey, CheckCellId, FindSortedCell, FindSortedBlock,
//          ComputeBigCellAtSizeLevel, FindSortedCellAt, CellCheck, CheckOverlapAt, CheckSiblingsAt, CheckSizeStepAt,
//          ToFailureKey, ToFailureStep, ToFailurePosition, ComputeFailureCells, ComputeSplitCellCountToAdd,
//          FindBlockNeighbours, ComputeCoveredCellState, ComputeCellStateBeside, ComputeFacePieceKind; and, for this
//          header's own functions and types only, ComputeBiasedCorner, ComputeSortKeyOfCorner, ComputeIsCellPresent,
//          ComputeBlockIdAt, FindSortedBlockAt, ComputeNeighbourEntry, ComputeStoredCellState,
//          ComputeLocalCoordInNeighbour, ComputeBlockSide, ComputeIsCoveredByActiveBigCell, and the struct LocalPlace
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), GridError (grid/grid_error.h), CellId, CellBlockId,
//          CellBlockIndex, CellPlaceIndex, FacePlaceIndex, CellCoord, SizeLevel, kBlockWidthInCells, kMaxSizeLevels,
//          kCellPlacesPerBlock, kFacePlacesPerBlock, kNoIndex, ComesBefore, ComputeIsCoordInRange, ToCellId,
//          ToCellCoord, ToSizeLevel, ToCellBlockIndex, ComputeBigCell, ComputeCellEighth, ComputeBiggerCellCoord and
//          the id bit layout (grid/grid_ids.h), GridFrame, GridTables, CellState, FacePieceKind (grid/grid_tables.h)
// Used by: the grid's CPU builder, its GPU half and the counting of a list (grid/); private to grid/
// Read:    docs/method-paper/fluidity-method.typ <sec-parallel-build>, <proc-build-grid-in-parallel>,
//          <proc-build-grid>, <sec-cell-blocks>, <sec-face-pieces>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// The passes of a sort key: 0 for x with the size level, 1 for y, 2 for z, run in that order.
inline constexpr int kSortKeyPassCount = 3;

// Computes biased(c, l) = (c + 2^(18 + l)) << (15 - l) for a low corner c in cells of size level l.
//
// Method: the sum is formed in unsigned 64-bit arithmetic, where it wraps to its true value, which is 0 or more for
//   every low corner of -2^18 or more; only that non-negative value is shifted.
// Requires: low_corner >= -2^18.
FLUIDITY_CPU_AND_GPU constexpr std::uint64_t
ComputeBiasedCorner(std::int32_t low_corner, SizeLevel size_level) noexcept {
  const auto size_level_number = static_cast<unsigned>(size_level);
  const std::uint64_t bias = std::uint64_t{1} << (18U + size_level_number);
  const std::uint64_t biased_sum = static_cast<std::uint64_t>(static_cast<std::int64_t>(low_corner)) + bias;
  return biased_sum << (15U - size_level_number);
}

// Computes one pass of the sort key of a low corner: (biased(x, l) << 4) | l, biased(y, l) or biased(z, l).
FLUIDITY_CPU_AND_GPU constexpr std::uint64_t
ComputeSortKeyOfCorner(CellCoord low_corner, SizeLevel size_level, int pass) noexcept {
  if (pass == 0) {
    return (ComputeBiasedCorner(low_corner.x, size_level) << 4U) | static_cast<std::uint64_t>(size_level);
  }
  if (pass == 1) {
    return ComputeBiasedCorner(low_corner.y, size_level);
  }
  return ComputeBiasedCorner(low_corner.z, size_level);
}

// Computes one pass of the key that sorts cells into the fixed order.
//
// Method: (biased(x, l) << 4) | l in pass 0, biased(y, l) in pass 1, biased(z, l) in pass 2 (see the file header).
// Order: the three passes of a stable order, pass 0 first, give the order of ComesBefore.
// Example: size level 0 {kMaxCoord, 0, 0} has (2^34 - 2^15) << 4 in pass 0; size level 0 {-kMaxCoord, 0, 0} has
//   2^15 << 4. CellId{0} has 2^64 - 1 in every pass.
// Requires: pass is 0, 1 or 2.
// Read: docs/method-paper/fluidity-method.typ <sec-parallel-build>
FLUIDITY_CPU_AND_GPU constexpr std::uint64_t ToCellSortKey(CellId cell, int pass) noexcept {
  assert(pass >= 0 && pass < kSortKeyPassCount);
  if ((cell.value & kIdCellMark) == 0) {
    return ~std::uint64_t{0};
  }
  return ComputeSortKeyOfCorner(ToCellCoord(cell), ToSizeLevel(cell), pass);
}

// Computes one pass of the key that sorts cells by their cell block, the blocks in the fixed order.
//
// Method: the key of ToCellSortKey with the block's low corner, 4 floor(c / 4) on each axis, in place of the cell's
//   coordinate c. Every cell of one block has the same key.
// Example: size level 0 {-kMaxCoord, -kMaxCoord, -kMaxCoord} lies in the lowest block of size level 0, low corner
//   -2^18 on each axis: 0 in every pass. CellId{0} has 2^64 - 1 in every pass.
// Requires: pass is 0, 1 or 2.
// Read: docs/method-paper/fluidity-method.typ <sec-parallel-build>
FLUIDITY_CPU_AND_GPU constexpr std::uint64_t ToBlockSortKey(CellId cell, int pass) noexcept {
  assert(pass >= 0 && pass < kSortKeyPassCount);
  if ((cell.value & kIdCellMark) == 0) {
    return ~std::uint64_t{0};
  }
  const CellCoord cell_coord = ToCellCoord(cell);
  const CellCoord block_low_corner = {
      .x = kBlockWidthInCells * ComputeBiggerCellCoord(cell_coord.x, 2),
      .y = kBlockWidthInCells * ComputeBiggerCellCoord(cell_coord.y, 2),
      .z = kBlockWidthInCells * ComputeBiggerCellCoord(cell_coord.z, 2),
  };
  return ComputeSortKeyOfCorner(block_low_corner, ToSizeLevel(cell), pass);
}

// Checks one input id against a frame: method paper <proc-build-grid> step 1 for one id.
//
// Order: the form of the id first, then its size level, then its coordinates.
// Example: with 3 size levels, size level 2 {kMaxCoord, -kMaxCoord, 0} is kNone; size level 3 {0, 0, 0} is
//   kBadInput; size level 1 {kMaxCoord + 1, 0, 0} is kCoordOutOfRange.
// Fails: kBadInput        bit 63 clear, so the value is no cell id; bits 61..62 not zero, which no cell id has; or
//                         a size level of size_level_count or more, which the frame does not use
//        kCoordOutOfRange a coordinate is kMaxCoord + 1, the one value outside the range that an id can hold
//        kNone otherwise.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>
FLUIDITY_CPU_AND_GPU constexpr GridError CheckCellId(GridFrame frame, CellId cell) noexcept {
  constexpr std::uint64_t kAxisBits = std::uint64_t{3} << kIdAxisShift;
  const bool has_cell_mark = (cell.value & kIdCellMark) != 0;
  const bool has_axis_bits = (cell.value & kAxisBits) != 0;
  if (!has_cell_mark || has_axis_bits) {
    return GridError::kBadInput;
  }
  if (static_cast<int>(ToSizeLevel(cell)) >= static_cast<int>(frame.size_level_count)) {
    return GridError::kBadInput;
  }
  if (!ComputeIsCoordInRange(ToCellCoord(cell))) {
    return GridError::kCoordOutOfRange;
  }
  return GridError::kNone;
}

// Finds a cell in a list sorted in the fixed order.
//
// Method: binary search with ComesBefore for the first entry that does not come before `cell`, then a comparison of
//   the ids.
// Order: of several equal entries, the first.
// Example: in {size level 0 {0,0,0}, size level 2 {4,4,4}}, size level 2 {4,4,4} is at 1 and size level 1 {0,0,0}
//   gives kNoIndex.
// Requires: sorted_cells holds count ids in the fixed order; it may be null when count is 0.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
FindSortedCell(const CellId* sorted_cells, std::uint32_t count, CellId cell) noexcept {
  assert(sorted_cells != nullptr || count == 0);
  std::uint32_t low = 0;
  std::uint32_t high = count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) / 2);
    if (ComesBefore(sorted_cells[middle], cell)) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  const bool is_found = low < count && sorted_cells[low].value == cell.value;
  return is_found ? low : kNoIndex;
}

// Finds a cell block in the block ids of a grid, sorted in the fixed order.
//
// Method: binary search with ComesBefore, as FindSortedCell.
// Example: in the blocks {size level 0 (0,0,0), size level 1 (0,0,0)}, size level 1 (0,0,0) is at index 1.
// Requires: sorted_blocks holds count ids in the fixed order; it may be null when count is 0.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellBlockIndex
FindSortedBlock(const CellBlockId* sorted_blocks, std::uint32_t count, CellBlockId block) noexcept {
  assert(sorted_blocks != nullptr || count == 0);
  std::uint32_t low = 0;
  std::uint32_t high = count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) / 2);
    if (ComesBefore(sorted_blocks[middle], block)) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  const bool is_found = low < count && sorted_blocks[low].value == block.value;
  return {.value = is_found ? low : kNoIndex};
}

// Computes the cell of size_level that holds `cell`: each coordinate divided by 2^(the cell's size level -
// size_level), rounded toward minus infinity.
//
// Method: halves one size level at a time with ComputeBiggerCellCoord, at most 15 times.
// Example: size level 3 {-5, 6, 7} at size level 1 is {-2, 1, 1}; size level 15 {kMaxCoord, -kMaxCoord, -1} at size
//   level 0 is {7, -8, -1}.
// Requires: size_level is at most the size level of `cell`; `cell` is a cell id whose coordinates are
//   -kMaxCoord..kMaxCoord + 1.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-sizes>
FLUIDITY_CPU_AND_GPU constexpr CellId ComputeBigCellAtSizeLevel(CellId cell, SizeLevel size_level) noexcept {
  const auto cell_size_level = static_cast<int>(ToSizeLevel(cell));
  const auto big_size_level = static_cast<int>(size_level);
  assert(big_size_level <= cell_size_level);
  CellCoord big_coord = ToCellCoord(cell);
  for (int halving = big_size_level; halving < cell_size_level; ++halving) {
    big_coord = {
        .x = ComputeBiggerCellCoord(big_coord.x, 1),
        .y = ComputeBiggerCellCoord(big_coord.y, 1),
        .z = ComputeBiggerCellCoord(big_coord.z, 1),
    };
  }
  return ToCellId(size_level, big_coord);
}

// Finds the cell of size_level at a raw coordinate in a sorted list: the one lookup for a place beside a cell.
//
// Method: a coordinate outside -kMaxCoord..kMaxCoord holds no cell of its size level, so it gives kNoIndex before
//   any id is made: packed, it would be the id of another cell. Otherwise FindSortedCell of its id.
// Example: size level 0 {-kMaxCoord - 1, 0, 0} gives kNoIndex even in a list holding size level 0
//   {kMaxCoord + 1, 0, 0}, whose id the packing would give.
// Requires: as FindSortedCell; size_level < kMaxSizeLevels.
// Read: docs/method-paper/fluidity-method.typ <proc-count-grid>
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
FindSortedCellAt(const CellId* sorted_cells, std::uint32_t count, SizeLevel size_level, CellCoord coord) noexcept {
  if (!ComputeIsCoordInRange(coord)) {
    return kNoIndex;
  }
  return FindSortedCell(sorted_cells, count, ToCellId(size_level, coord));
}

// What one check of one sorted position found. A default CellCheck is no failure.
struct CellCheck {
  GridError error = GridError::kNone;
  CellId cell{};  // The failing cell: the smaller cell of a pair, or the missing sibling. CellId{0} when none.
  CellId other_cell{};  // The bigger cell of an overlap or a size step, or the big cell of a missing sibling.
  // The low 8 bits of the failure key: (walk step << 3) | eighth for a missing sibling, the direction 0..25 for the
  // size step rule, 0 otherwise.
  std::uint8_t detail = 0;
};

// Checks the cell at one position of a sorted list for an overlap: method paper <proc-build-grid> step 2.
//
// Method: the cell overlaps when it equals the entry before it, or when an active cell of a bigger size level holds
//   it; the bigger cells are looked up one size level up first. Only the smaller cell of a pair reports, and of two
//   equal entries only the second.
// Example: in {size level 0 {-1,0,0}, size level 2 {-3,1,2}}, position 1 gives kCellsOverlap, cell size level 2
//   {-3,1,2}, other_cell size level 0 {-1,0,0}; position 0 gives no failure.
// Requires: position < count; the list is in the fixed order and every id passed CheckCellId.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <proc-build-grid-in-parallel>
FLUIDITY_CPU_AND_GPU constexpr CellCheck
CheckOverlapAt(const CellId* sorted_cells, std::uint32_t count, std::uint32_t position) noexcept {
  assert(sorted_cells != nullptr && position < count);
  const CellId cell = sorted_cells[position];
  if (position > 0 && sorted_cells[position - 1].value == cell.value) {
    return {.error = GridError::kCellsOverlap, .cell = cell, .other_cell = cell, .detail = 0};
  }
  // Halving a coordinate in range keeps it in range, so every bigger cell has an id.
  for (int big_size_level = static_cast<int>(ToSizeLevel(cell)) - 1; big_size_level >= 0; --big_size_level) {
    const CellId big_cell = ComputeBigCellAtSizeLevel(cell, static_cast<SizeLevel>(big_size_level));
    if (FindSortedCell(sorted_cells, count, big_cell) != kNoIndex) {
      return {.error = GridError::kCellsOverlap, .cell = cell, .other_cell = big_cell, .detail = 0};
    }
  }
  return {};
}

// Says whether a small cell is present for the sibling check: active, or split with an active cell at the end of its
// eighth-0 chain. The chain is followed on raw coordinates, 2 c on each axis, and stops at a coordinate outside the
// range, where no smaller cell can exist, or after size level 15.
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsCellPresent(const CellId* sorted_cells, std::uint32_t count, SizeLevel size_level, CellCoord coord) noexcept {
  CellCoord chain_coord = coord;
  for (int chain_size_level = static_cast<int>(size_level); chain_size_level < kMaxSizeLevels; ++chain_size_level) {
    if (!ComputeIsCoordInRange(chain_coord)) {
      return false;
    }
    if (FindSortedCellAt(sorted_cells, count, static_cast<SizeLevel>(chain_size_level), chain_coord) != kNoIndex) {
      return true;
    }
    // In range, |c| <= 2^18 - 1, so 2 c fits a 32-bit integer.
    chain_coord = {.x = 2 * chain_coord.x, .y = 2 * chain_coord.y, .z = 2 * chain_coord.z};
  }
  return false;
}

// Checks the siblings of the cell at one position of a sorted list, and of every split cell it adds: method paper
// <proc-build-grid> step 3.
//
// Method: let d be the cell. While d is not of size level 0, every one of the 8 small cells of d's big cell must be
//   present (active, or split with an active cell at the end of its eighth-0 chain); the first that is not, in eighth
//   order, is the failure. Then, when d is eighth 0 of its big cell, d becomes that big cell, the split cell
//   ComputeSplitCellCountToAdd adds, and the walk goes on; otherwise it stops. In a valid list each split cell has its
//   siblings checked once, at the active cell at the end of its eighth-0 chain.
//   Why this finds every missing cell. Call a cell missing when its big cell holds an active cell but the cell holds
//   none. Take a smallest missing cell g, of big cell B. Some small cell s of B holds an active cell. If s is not
//   active, follow its eighth-0 chain: each cell on it holds an active cell, since one that did not would be a smaller
//   missing cell than g, so the chain ends at an active cell a. The walk from a climbs the chain, whose cells are all
//   eighth 0, to s and checks the small cells of B. There g is not present: it is not active, and no cell of its chain
//   is. So the walk from a reports a failure at B. A cell is reported only when it is not active and no cell of its
//   eighth-0 chain is: then that cell, or the first cell of its chain that holds no active cell, is missing, and the
//   list is bad.
// Example: the 7 size-level-1 cells of size level 0 {-1,0,0} without eighth 5 give, at every position,
//   kSiblingCellMissing with cell size level 1 {-1,0,1}, other_cell size level 0 {-1,0,0}, detail 5.
//   Coordinate limit: a small cell outside the range cannot exist, so a cell can be split only when its coordinate
//   is -(2^17 - 1)..2^17 - 1 on every axis, where its small cells 2 c and 2 c + 1 stay within +-kMaxCoord. A cell of
//   size level 1 or more at -kMaxCoord lies in a big cell at -2^17, whose eighth-0 small cell would be at -2^18: such a
//   list is never valid, and the check fails with cell CellId{0}, since no id names that cell, other_cell the big
//   cell, and detail 0. In general a cell of size level l at c holds cells of size level m only when one of its cells
//   of size level m - 1, c 2^(m - 1 - l) to (c + 1) 2^(m - 1 - l) - 1, can be split. Example: a size-level-0 cell can
//   hold size-level-15 cells only at -8..7 on every axis; at -8 not at its low corner, whose size-level-14 cell -2^17
//   cannot be split.
// Fails: kSiblingCellMissing  cell = the small cell that is not present, CellId{0} when its coordinate is outside the
//                             range; other_cell = its big cell; detail = (walk step << 3) | its eighth, walk step 0
//                             for the cell's own big cell. The walk step is at most 14, so the detail fits 8 bits.
// Requires: position < count; the list is in the fixed order, every id passed CheckCellId, and no cells overlap.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <proc-build-grid-in-parallel>
FLUIDITY_CPU_AND_GPU constexpr CellCheck
CheckSiblingsAt(const CellId* sorted_cells, std::uint32_t count, std::uint32_t position) noexcept {
  assert(sorted_cells != nullptr && position < count);
  CellId walk_cell = sorted_cells[position];
  for (unsigned walk_step = 0; ToSizeLevel(walk_cell) != SizeLevel{0}; ++walk_step) {
    const CellId big_cell = ComputeBigCell(walk_cell);
    const CellCoord big_coord = ToCellCoord(big_cell);
    const auto small_size_level = ToSizeLevel(walk_cell);
    for (unsigned cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
      const CellCoord small_coord = {
          .x = (2 * big_coord.x) + static_cast<std::int32_t>(cell_eighth & 1U),
          .y = (2 * big_coord.y) + static_cast<std::int32_t>((cell_eighth >> 1U) & 1U),
          .z = (2 * big_coord.z) + static_cast<std::int32_t>((cell_eighth >> 2U) & 1U),
      };
      if (!ComputeIsCellPresent(sorted_cells, count, small_size_level, small_coord)) {
        const bool is_small_coord_in_range = ComputeIsCoordInRange(small_coord);
        return {
            .error = GridError::kSiblingCellMissing,
            .cell = is_small_coord_in_range ? ToCellId(small_size_level, small_coord) : CellId{0},
            .other_cell = big_cell,
            .detail = static_cast<std::uint8_t>((walk_step << 3U) | cell_eighth),
        };
      }
    }
    if (ComputeCellEighth(walk_cell) != 0) {
      break;
    }
    walk_cell = big_cell;
  }
  return {};
}

// Checks the size step rule at the cell at one position of a sorted list: method paper <proc-build-grid> step 4.
//
// Method: for each of the 26 directions (dz, dy, dx), each -1, 0 or +1 and not all 0, in the order dz, then dy,
//   then dx, numbered 0..25 with the cell itself left out, takes the place of the cell's own size level beside it,
//   c + d on the raw coordinate. An active cell two or more size levels bigger that covers the place breaks the rule.
//   The covering coordinate at each bigger size level is the raw coordinate halved once per size level, looked up
//   with FindSortedCellAt; at the low limit the halved place is the cell's own big cell, so no cell lies beyond.
// Order: the first direction; in it, the nearest bigger size level first (only one active cell can cover a place).
// Example: size level 2 {4,4,4} beside size level 0 {0,0,0}: direction (-1, -1, -1) is 0, the place (3, 3, 3) of
//   size level 2 lies in size level 0 {0,0,0}. Size level 2 {kMaxCoord, 0, 0} beside size level 0 {65536, 0, 0}:
//   direction (0, 0, +1) is 13, the place x = 2^18 lies in size level 0 x = 2^16.
// Fails: kSizeStepRuleBroken  cell = the cell at the position; other_cell = the bigger cell; detail = the direction.
// Requires: position < count; the list is in the fixed order, every id passed CheckCellId, and no cells overlap.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <proc-build-grid-in-parallel>, <proc-count-grid>
FLUIDITY_CPU_AND_GPU constexpr CellCheck
CheckSizeStepAt(const CellId* sorted_cells, std::uint32_t count, std::uint32_t position) noexcept {
  assert(sorted_cells != nullptr && position < count);
  const CellId cell = sorted_cells[position];
  const auto cell_size_level = static_cast<int>(ToSizeLevel(cell));
  const CellCoord cell_coord = ToCellCoord(cell);
  std::uint8_t direction = 0;
  for (std::int32_t dz = -1; dz <= 1; ++dz) {
    for (std::int32_t dy = -1; dy <= 1; ++dy) {
      for (std::int32_t dx = -1; dx <= 1; ++dx) {
        if (dz == 0 && dy == 0 && dx == 0) {
          continue;
        }
        // The place beside the cell, -2^18..2^18 on each axis: ComputeBiggerCellCoord halves it exactly.
        CellCoord covering_coord = {.x = cell_coord.x + dx, .y = cell_coord.y + dy, .z = cell_coord.z + dz};
        for (int bigger_size_level = cell_size_level - 1; bigger_size_level >= 0; --bigger_size_level) {
          covering_coord = {
              .x = ComputeBiggerCellCoord(covering_coord.x, 1),
              .y = ComputeBiggerCellCoord(covering_coord.y, 1),
              .z = ComputeBiggerCellCoord(covering_coord.z, 1),
          };
          const bool is_two_or_more_bigger = bigger_size_level <= cell_size_level - 2;
          const auto covering_size_level = static_cast<SizeLevel>(bigger_size_level);
          if (is_two_or_more_bigger
              && FindSortedCellAt(sorted_cells, count, covering_size_level, covering_coord) != kNoIndex) {
            return {
                .error = GridError::kSizeStepRuleBroken,
                .cell = cell,
                .other_cell = ToCellId(covering_size_level, covering_coord),
                .detail = direction,
            };
          }
        }
        ++direction;
      }
    }
  }
  return {};
}

// Computes the failure key of one failing item: (step << 56) | (position << 8) | detail.
//
// Method: the step takes the high 8 bits, the position the next 48, the detail the low 8, so the smallest key is the
//   earliest step, then the smallest position, then the smallest detail. The steps: 1 the ids (position = input
//   index), 2 overlaps, 3 siblings, 4 the size step rule (position = place in the fixed cell order), 5 the active and
//   split cells against 64 max_cell_blocks, 6 the blocks against max_cell_blocks (position 0), 7 a block of size level
//   1 or more without its bigger block (position = the block's index).
// Example: ToFailureKey(2, 5, 0) = 144115188075857152, above ToFailureKey(1, 2^40, 255).
// Requires: step < 256; position < 2^48.
// Read: docs/method-paper/fluidity-method.typ <sec-parallel-build>
//
// clang-tidy: a step and a position; the swappable-parameters check flags the integers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU constexpr std::uint64_t
ToFailureKey(int step, std::uint64_t position, std::uint8_t detail) noexcept {
  assert(step >= 0 && step < 256);
  assert(position < (std::uint64_t{1} << 48U));
  return (static_cast<std::uint64_t>(step) << 56U) | (position << 8U) | detail;
}

// Gives the step of a failure key: its high 8 bits. 255 for kNoFailureKey.
//
// Example: ToFailureStep(ToFailureKey(6, 2^48 - 1, 255)) = 6.
// Read: docs/method-paper/fluidity-method.typ <sec-parallel-build>
FLUIDITY_CPU_AND_GPU constexpr int ToFailureStep(std::uint64_t failure_key) noexcept {
  return static_cast<int>(failure_key >> 56U);
}

// Gives the position of a failure key: bits 8..55.
//
// Example: ToFailurePosition(ToFailureKey(3, 7, 9)) = 7.
// Read: docs/method-paper/fluidity-method.typ <sec-parallel-build>
FLUIDITY_CPU_AND_GPU constexpr std::uint64_t ToFailurePosition(std::uint64_t failure_key) noexcept {
  return (failure_key >> 8U) & ((std::uint64_t{1} << 48U) - 1);
}

// Computes the error and the cells of a build's smallest failure key, once the minimum is known, so one thread can fill
// the status record: method paper <proc-build-grid-in-parallel> step P12.
//
// Method: step 1 runs CheckCellId again on the input id at the key's position, in input order, and names that id as
//   given, its raw value, even when bit 63 is clear. Steps 2, 3 and 4 run CheckOverlapAt, CheckSiblingsAt or
//   CheckSizeStepAt again at the key's position in the sorted list. Steps 5 and 6 give kTooManyCellBlocks with zero
//   cells. Step 7, a block of size level 1 or more without its bigger block, gives kInternalFault with zero cells: a
//   fault in the build, reported with no cell.
// Example: in {size level 0 {0,0,0}, size level 2 {4,4,4}}, ToFailureKey(4, 1, 0) gives kSizeStepRuleBroken, cell
//   size level 2 {4,4,4}, other_cell size level 0 {0,0,0}, detail 0. With CellId{0} at input index 3,
//   ToFailureKey(1, 3, 0) gives kBadInput with cell CellId{0}.
// Fails: kInternalFault  the key's step is 0 or above 7, its position is not below input_count (step 1) or count
//                        (steps 2 to 4), or the check finds no failure there; zero cells. A key that the build
//                        computed from these lists never has any of these, so each means a fault in the build. Step 7
//                        gives it too, as above.
// Requires: the lists are the ones the key's checks ran on; input_cells holds input_count ids in input order and
//   sorted_cells count ids in the fixed order; either may be null when its count is 0.
// Read: docs/method-paper/fluidity-method.typ <sec-parallel-build>, <proc-build-grid-in-parallel>
//
// clang-tidy: counts of cells and a failure key; the swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU constexpr CellCheck ComputeFailureCells(
    GridFrame frame,
    const CellId* input_cells,
    std::uint32_t input_count,
    const CellId* sorted_cells,
    std::uint32_t count,
    std::uint64_t failure_key
) noexcept {
  const int step = ToFailureStep(failure_key);
  const std::uint64_t position = ToFailurePosition(failure_key);
  const CellCheck internal_fault = {.error = GridError::kInternalFault};
  if (step == 5 || step == 6) {
    return {.error = GridError::kTooManyCellBlocks};
  }
  if (step == 1) {
    if (position >= input_count) {
      return internal_fault;
    }
    const CellId input_cell = input_cells[position];
    const GridError id_error = CheckCellId(frame, input_cell);
    if (id_error == GridError::kNone) {
      return internal_fault;
    }
    return {.error = id_error, .cell = input_cell};
  }
  const bool is_sorted_check = step >= 2 && step <= 4;
  if (!is_sorted_check || position >= count) {
    return internal_fault;
  }
  const auto sorted_position = static_cast<std::uint32_t>(position);
  CellCheck cell_check{};
  if (step == 2) {
    cell_check = CheckOverlapAt(sorted_cells, count, sorted_position);
  } else if (step == 3) {
    cell_check = CheckSiblingsAt(sorted_cells, count, sorted_position);
  } else {
    cell_check = CheckSizeStepAt(sorted_cells, count, sorted_position);
  }
  return cell_check.error == GridError::kNone ? internal_fault : cell_check;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Computes how many split cells an active cell adds: method paper <proc-build-grid-in-parallel> step P7.
//
// Method: the cell adds its big cell when it is eighth 0 of it, then that cell's big cell when that one is eighth 0
//   too, and so on: a loop of at most 15 steps, with no search. The cells added are ComputeBigCellAtSizeLevel(cell, l -
//   1), ComputeBigCellAtSizeLevel(cell, l - 2), ..., for l the cell's size level. Since siblings exist together once
//   step 3 has passed, every split cell is the big cell of exactly one eighth-0 small cell, and so added exactly once.
// Example: size level 15 {0,0,0} adds 15 cells; size level 3 {2,0,0} adds size level 2 {1,0,0}, which is eighth 1, so
//   1; size level 15 {kMaxCoord, 0, 0}, odd, adds 0.
// Requires: `cell` is a cell id.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <proc-build-grid-in-parallel>
FLUIDITY_CPU_AND_GPU constexpr int ComputeSplitCellCountToAdd(CellId cell) noexcept {
  int added_count = 0;
  CellId chain_cell = cell;
  while (ToSizeLevel(chain_cell) != SizeLevel{0} && ComputeCellEighth(chain_cell) == 0) {
    ++added_count;
    chain_cell = ComputeBigCell(chain_cell);
  }
  return added_count;
}

// Computes the id of the block of size_level at a block coordinate: the same bits as ToCellBlockId gives for any cell
// of that block. A block id packs its block coordinate as a cell id packs a cell coordinate.
// Requires: size_level < kMaxSizeLevels; each coordinate within -kMaxCoord..kMaxCoord + 1.
FLUIDITY_CPU_AND_GPU constexpr CellBlockId ComputeBlockIdAt(SizeLevel size_level, CellCoord block_coord) noexcept {
  return {.value = ToCellId(size_level, block_coord).value};
}

// Finds the block of size_level at a raw block coordinate in the sorted block ids: the one lookup for a block beside
// a block, above it or below it. A coordinate outside -kMaxCoord..kMaxCoord holds no block and gives kNoIndex before
// any id is made, since its packed id would name another block.
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t FindSortedBlockAt(
    const CellBlockId* sorted_blocks,
    std::uint32_t count,
    SizeLevel size_level,
    CellCoord block_coord
) noexcept {
  if (!ComputeIsCoordInRange(block_coord)) {
    return kNoIndex;
  }
  return FindSortedBlock(sorted_blocks, count, ComputeBlockIdAt(size_level, block_coord)).value;
}

// Computes the entry 0..26 of the offset (dx, dy, dz), each -1, 0 or +1, in a block's 27 same-size-level neighbours:
// ((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1). The offset (0, 0, 0) is entry 13.
FLUIDITY_CPU_AND_GPU constexpr int ComputeNeighbourEntry(int dx, int dy, int dz) noexcept {
  return ((((dz + 1) * 3) + (dy + 1)) * 3) + (dx + 1);
}

// Finds the 27 same-size-level neighbour blocks, the bigger block and the 8 smaller blocks of one block, and writes
// them to its entries of the three tables: method paper <proc-build-grid> step 6,
// <proc-build-grid-in-parallel> step P9.
//
// Method: entry 13 is the block itself; every other entry is FindSortedBlock of the block id at the entry's block
//   coordinate. For a block of size level l at block coordinate b, the neighbour at offset d is at b + d (size level
//   l), the bigger block at floor(b / 2) (size level l - 1), and the smaller block of eighth e at 2 b + (e & 1,
//   (e >> 1) & 1, (e >> 2) & 1) (size level l + 1). Each eighth of a block is 2 x 2 x 2 of its cells, whose small
//   cells fill exactly one block of size level l + 1. A block coordinate outside -kMaxCoord..kMaxCoord has no block,
//   decided before any id is made. A size-level-0 block has no bigger block, and a size-level-15 block no smaller
//   blocks. The entries go where the layouts of GridTables (grid/grid_tables.h) put them. When the block of size
//   level l - 1 at floor(b / 2) is not in the list, bigger_block gets kNoIndex; a build treats that, for a block of
//   size level 1 or more, as an internal fault, since every such block lies in a split cell whose block exists.
// Example: the blocks {size level 0 (0,0,0), size level 1 (0,0,0)}: block 1 has bigger block 0; block 0 has smaller
//   block 1 for eighth 0 and kNoIndex for eighths 1..7; entry 13 of each is the block itself, every other
//   neighbour entry kNoIndex.
// Requires: block_ids holds block_count ids in the fixed order, each once, every coordinate within
//   -kMaxCoord..kMaxCoord + 1; block < block_count; each table has room for block_count entries of the block.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <proc-build-grid-in-parallel>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr void FindBlockNeighbours(
    const CellBlockId* block_ids,
    std::uint32_t block_count,
    CellBlockIndex block,
    std::uint32_t* same_size_level_blocks,
    std::uint32_t* bigger_block,
    std::uint32_t* smaller_blocks
) noexcept {
  assert(block_ids != nullptr && block.value < block_count);
  assert(same_size_level_blocks != nullptr && bigger_block != nullptr && smaller_blocks != nullptr);
  // A block id has the bits of the cell id at its block coordinate, so ToCellCoord and ToSizeLevel read it back.
  const CellId block_as_cell = {.value = block_ids[block.value].value};
  const CellCoord block_coord = ToCellCoord(block_as_cell);
  const SizeLevel size_level = ToSizeLevel(block_as_cell);
  const auto size_level_number = static_cast<int>(size_level);
  const std::uint64_t block_number = block.value;

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const auto entry = static_cast<std::uint64_t>(ComputeNeighbourEntry(dx, dy, dz));
        const bool is_self = dx == 0 && dy == 0 && dz == 0;
        const CellCoord neighbour_coord = {.x = block_coord.x + dx, .y = block_coord.y + dy, .z = block_coord.z + dz};
        same_size_level_blocks[(block_number * 27) + entry] =
            is_self ? block.value : FindSortedBlockAt(block_ids, block_count, size_level, neighbour_coord);
      }
    }
  }

  bigger_block[block_number] = kNoIndex;
  if (size_level_number >= 1) {
    const CellCoord bigger_coord = {
        .x = ComputeBiggerCellCoord(block_coord.x, 1),
        .y = ComputeBiggerCellCoord(block_coord.y, 1),
        .z = ComputeBiggerCellCoord(block_coord.z, 1),
    };
    const auto bigger_size_level = static_cast<SizeLevel>(size_level_number - 1);
    bigger_block[block_number] = FindSortedBlockAt(block_ids, block_count, bigger_size_level, bigger_coord);
  }

  for (unsigned cell_eighth = 0; cell_eighth < 8; ++cell_eighth) {
    std::uint32_t smaller = kNoIndex;
    if (size_level_number < kMaxSizeLevels - 1) {
      // Every coordinate is at most 2^18 in size, so 2 b + 1 fits a 32-bit integer.
      const CellCoord smaller_coord = {
          .x = (2 * block_coord.x) + static_cast<std::int32_t>(cell_eighth & 1U),
          .y = (2 * block_coord.y) + static_cast<std::int32_t>((cell_eighth >> 1U) & 1U),
          .z = (2 * block_coord.z) + static_cast<std::int32_t>((cell_eighth >> 2U) & 1U),
      };
      const auto smaller_size_level = static_cast<SizeLevel>(size_level_number + 1);
      smaller = FindSortedBlockAt(block_ids, block_count, smaller_size_level, smaller_coord);
    }
    smaller_blocks[(block_number * 8) + cell_eighth] = smaller;
  }
}

// The cell place given by local coordinates, x, y and z, of a block: 0..3 inside the block, and on at most one axis
// -1 or 4, one place beyond the block's low or high side.
struct LocalPlace {
  int x;
  int y;
  int z;
};

// Computes the stored state of local place (x, y, z), each 0..3, of a block that exists.
FLUIDITY_CPU_AND_GPU constexpr CellState
ComputeStoredCellState(const GridTables& tables, std::uint32_t block, LocalPlace place) noexcept {
  const int place_in_block = (((place.z * 4) + place.y) * 4) + place.x;
  const std::uint64_t block_start = static_cast<std::uint64_t>(block) * kCellPlacesPerBlock;
  return tables.cell_states[block_start + static_cast<std::uint64_t>(place_in_block)];
}

// Computes the local coordinate of a place one cell beyond a block side in the block on that side: -1 is 3 of the
// block below, 4 is 0 of the block above, 0..3 stay.
FLUIDITY_CPU_AND_GPU constexpr int ComputeLocalCoordInNeighbour(int local_coord) noexcept {
  if (local_coord < 0) {
    return 3;
  }
  if (local_coord > 3) {
    return 0;
  }
  return local_coord;
}

// Computes the side, -1, 0 or +1, on which a local coordinate lies beyond a block: -1 below 0, +1 above 3.
FLUIDITY_CPU_AND_GPU constexpr int ComputeBlockSide(int local_coord) noexcept {
  if (local_coord < 0) {
    return -1;
  }
  if (local_coord > 3) {
    return 1;
  }
  return 0;
}

// Says whether an active cell one size level bigger covers a local place of `block` (each local coordinate 0..3, or
// -1 or 4 on one axis), through the bigger block.
//
// Method: on each axis, a block at block coordinate b is the low half (b even) or the high half (b odd) of its bigger
//   block, so its local coordinate c is the bigger block's 2 (b & 1) + floor(c / 2): 0..3 inside, -1 or 4 beyond.
//   The halving of -1 is -1, taken through ComputeBiggerCellCoord. Beyond, the covering place lies in the bigger
//   block's same-size-level neighbour on that side, at local 3 or 0. No bigger block, or no such neighbour, covers
//   nothing. Only a kActive cover counts: a place marked kInsideBiggerCell covers nothing one size level down.
FLUIDITY_CPU_AND_GPU constexpr bool
ComputeIsCoveredByActiveBigCell(const GridTables& tables, std::uint32_t block, LocalPlace place) noexcept {
  const std::uint32_t bigger_block = tables.bigger_block[block];
  if (bigger_block == kNoIndex) {
    return false;
  }
  const CellCoord block_coord = ToCellCoord(CellId{.value = tables.block_ids[block].value});
  // The low bit of a coordinate as unsigned is its parity, for a negative coordinate too.
  const auto half_x = static_cast<std::int32_t>(static_cast<std::uint32_t>(block_coord.x) & 1U);
  const auto half_y = static_cast<std::int32_t>(static_cast<std::uint32_t>(block_coord.y) & 1U);
  const auto half_z = static_cast<std::int32_t>(static_cast<std::uint32_t>(block_coord.z) & 1U);
  const LocalPlace covering_place = {
      .x = (2 * half_x) + ComputeBiggerCellCoord(place.x, 1),
      .y = (2 * half_y) + ComputeBiggerCellCoord(place.y, 1),
      .z = (2 * half_z) + ComputeBiggerCellCoord(place.z, 1),
  };
  const int side_x = ComputeBlockSide(covering_place.x);
  const int side_y = ComputeBlockSide(covering_place.y);
  const int side_z = ComputeBlockSide(covering_place.z);
  std::uint32_t covering_block = bigger_block;
  if (side_x != 0 || side_y != 0 || side_z != 0) {
    const auto entry = static_cast<std::uint64_t>(ComputeNeighbourEntry(side_x, side_y, side_z));
    covering_block = tables.same_size_level_blocks[(static_cast<std::uint64_t>(bigger_block) * 27) + entry];
    if (covering_block == kNoIndex) {
      return false;
    }
  }
  const LocalPlace place_in_covering_block = {
      .x = ComputeLocalCoordInNeighbour(covering_place.x),
      .y = ComputeLocalCoordInNeighbour(covering_place.y),
      .z = ComputeLocalCoordInNeighbour(covering_place.z),
  };
  return ComputeStoredCellState(tables, covering_block, place_in_covering_block) == CellState::kActive;
}

// Computes the state of a cell place that holds neither an active nor a split cell: method paper <proc-build-grid>
// step 7, <proc-build-grid-in-parallel> step P10 (c).
//
// Method: kInsideBiggerCell when the place of the bigger block that covers it is kActive, else kAbsent. A block of
//   size level l at block coordinate b is the low or the high half of its bigger block on each axis, so its local
//   coordinate c is the bigger block's 2 (b & 1) + floor(c / 2). A block with no bigger block, size level 0, gives
//   kAbsent. It reads only kActive marks, and no call of this step writes one, so the places may be filled in any
//   order, or all at once.
// Example: in the blocks {size level 0 (0,0,0), size level 1 (0,0,0)} with size level 0 {0,0,0} active and {1,0,0}
//   split, the size-level-1 place (1, 0, 1) is kInsideBiggerCell, covered by place (0, 0, 0); the place (0, 2, 0) is
//   kAbsent, since its covering place (0, 1, 0) holds no active cell.
// Requires: `cell` is a place of block 0..block_count - 1 whose stored state is neither kActive nor kSplit; the
//   block ids, the neighbour tables, the bigger blocks and every kActive mark are written.
// Read: docs/method-paper/fluidity-method.typ <proc-build-grid>, <proc-build-grid-in-parallel>, <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellState
ComputeCoveredCellState(const GridTables& tables, CellPlaceIndex cell) noexcept {
  const std::uint32_t block = ToCellBlockIndex(cell).value;
  assert(block < tables.block_count);
  assert(tables.cell_states[cell.value] != CellState::kActive && tables.cell_states[cell.value] != CellState::kSplit);
  const auto place_in_block = static_cast<int>(cell.value % static_cast<std::uint32_t>(kCellPlacesPerBlock));
  const LocalPlace place = {
      .x = place_in_block % 4,
      .y = (place_in_block / 4) % 4,
      .z = place_in_block / 16,
  };
  return ComputeIsCoveredByActiveBigCell(tables, block, place) ? CellState::kInsideBiggerCell : CellState::kAbsent;
}

// Computes the state of a cell place of the size level of `block`, given by local coordinates of which one may lie one
// cell beyond the block, -1 or 4.
//
// Method: inside the block, the stored state. Beyond it, when the same-size-level neighbour block on that side
//   exists, the stored state of its place, at local 3 or 0. When it does not, no cell of this size level is there:
//   the place is kInsideBiggerCell when an active cell one size level bigger covers it, else kAbsent. The cover is
//   found through the bigger block: `block` is the low or the high half of it along the axis; on the half's outer
//   side the cover lies in the bigger block's same-size-level neighbour on that side (kAbsent when there is none),
//   on its inner side in the bigger block itself. A size-level-0 block has no bigger block: kAbsent.
//   Coordinate limits: a neighbour beyond -kMaxCoord..kMaxCoord has no entry in the tables, so nothing lies beyond the
//   low limit, and beyond the high limit only a bigger cell, found at its own size level through the bigger block.
// Example: with size level 0 {-1,0,0} active and the 8 size-level-1 cells of size level 0 {0,0,0}, the
//   size-level-1 block (0,0,0) has no size-level-1 neighbour on its low X side; local (-1, 0, 0) is the size-level-1
//   place x = -1, covered by size level 0 {-1,0,0}: kInsideBiggerCell. Local (0, -1, 0) lies under the bigger block's
//   low Y side, where no block is: kAbsent.
// Requires: block < block_count; each local coordinate is -1..4, and at most one of them is outside 0..3; every cell
//   state of the grid is written.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>, <sec-cell-blocks>
//
// clang-tidy: three local coordinates in x, y, z order; the swappable-parameters check flags the integers.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
FLUIDITY_CPU_AND_GPU constexpr CellState
ComputeCellStateBeside(const GridTables& tables, CellBlockIndex block, int local_x, int local_y, int local_z) noexcept {
  assert(block.value < tables.block_count);
  const int side_x = ComputeBlockSide(local_x);
  const int side_y = ComputeBlockSide(local_y);
  const int side_z = ComputeBlockSide(local_z);
  const bool is_at_most_one_place_beyond =
      local_x >= -1 && local_x <= 4 && local_y >= -1 && local_y <= 4 && local_z >= -1 && local_z <= 4;
  const int beyond_count = (side_x != 0 ? 1 : 0) + (side_y != 0 ? 1 : 0) + (side_z != 0 ? 1 : 0);
  assert(is_at_most_one_place_beyond && beyond_count <= 1);
  static_cast<void>(is_at_most_one_place_beyond);
  const LocalPlace place = {.x = local_x, .y = local_y, .z = local_z};
  if (beyond_count == 0) {
    return ComputeStoredCellState(tables, block.value, place);
  }
  const auto entry = static_cast<std::uint64_t>(ComputeNeighbourEntry(side_x, side_y, side_z));
  const std::uint32_t neighbour = tables.same_size_level_blocks[(static_cast<std::uint64_t>(block.value) * 27) + entry];
  if (neighbour != kNoIndex) {
    const LocalPlace place_in_neighbour = {
        .x = ComputeLocalCoordInNeighbour(local_x),
        .y = ComputeLocalCoordInNeighbour(local_y),
        .z = ComputeLocalCoordInNeighbour(local_z),
    };
    return ComputeStoredCellState(tables, neighbour, place_in_neighbour);
  }
  return ComputeIsCoveredByActiveBigCell(tables, block.value, place) ? CellState::kInsideBiggerCell
                                                                     : CellState::kAbsent;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

// Computes what one face place holds: method paper <proc-build-grid> step 8, <proc-build-grid-in-parallel> step P11.
//
// Method: the face place's block, axis and local place come from its index (layout of GridTables). Plane p lies
//   between the local cells p - 1 (low) and p (high) along the axis; both states come from ComputeCellStateBeside.
//   Plane 4 is kNotAPiece when the same-size-level block on the high side exists, since the face piece is kept at
//   that block's plane 0. Otherwise: active and active, or active and kInsideBiggerCell (either side) give
//   kBetweenCells; active on the high side and kAbsent on the low give kGridEdgeLow; active on the low side and
//   kAbsent on the high give kGridEdgeHigh; anything else, active beside kSplit or no active side, gives kNotAPiece.
//   So a face piece is kept once, at the size level of its smaller cell (<sec-face-pieces>).
// Example: size level 0 {0,0,0} active and the 8 size-level-1 cells of size level 0 {1,0,0}: block 0 of size level 0
//   and block 1 of size level 1. Plane 2 of block 1 along X lies between size-level-1 local cells 1 (inside the big
//   cell: kInsideBiggerCell) and 2 (active), so its 4 places beside the active cells are kBetweenCells. Plane 1 of
//   block 0 along X lies between the active big cell and the split cell: kNotAPiece.
// Requires: `face` is a face place of block 0..block_count - 1; every cell state of the grid is written.
// Read: docs/method-paper/fluidity-method.typ <sec-face-pieces>, <proc-build-grid>, <proc-build-grid-in-parallel>
FLUIDITY_CPU_AND_GPU constexpr FacePieceKind
ComputeFacePieceKind(const GridTables& tables, FacePlaceIndex face) noexcept {
  const CellBlockIndex block = ToCellBlockIndex(face);
  assert(block.value < tables.block_count);
  const auto place_in_block = static_cast<int>(face.value % static_cast<std::uint32_t>(kFacePlacesPerBlock));
  const int axis_number = place_in_block / 80;
  const int place_on_axis = place_in_block % 80;
  // The place's local coordinates; along the axis the plane, 0..4, the low side of local cell p.
  LocalPlace high_place{};
  LocalPlace low_place{};
  int plane = 0;
  if (axis_number == 0) {
    high_place = {.x = place_on_axis % 5, .y = (place_on_axis / 5) % 4, .z = place_on_axis / 20};
    plane = high_place.x;
    low_place = {.x = plane - 1, .y = high_place.y, .z = high_place.z};
  } else if (axis_number == 1) {
    high_place = {.x = place_on_axis % 4, .y = (place_on_axis / 4) % 5, .z = place_on_axis / 20};
    plane = high_place.y;
    low_place = {.x = high_place.x, .y = plane - 1, .z = high_place.z};
  } else {
    high_place = {.x = place_on_axis % 4, .y = (place_on_axis / 4) % 4, .z = place_on_axis / 16};
    plane = high_place.z;
    low_place = {.x = high_place.x, .y = high_place.y, .z = plane - 1};
  }

  if (plane == 4) {
    const int side_x = axis_number == 0 ? 1 : 0;
    const int side_y = axis_number == 1 ? 1 : 0;
    const int side_z = axis_number == 2 ? 1 : 0;
    const auto entry = static_cast<std::uint64_t>(ComputeNeighbourEntry(side_x, side_y, side_z));
    const std::uint32_t high_block =
        tables.same_size_level_blocks[(static_cast<std::uint64_t>(block.value) * 27) + entry];
    if (high_block != kNoIndex) {
      return FacePieceKind::kNotAPiece;
    }
  }

  const CellState low_state = ComputeCellStateBeside(tables, block, low_place.x, low_place.y, low_place.z);
  const CellState high_state = ComputeCellStateBeside(tables, block, high_place.x, high_place.y, high_place.z);
  const bool is_low_active = low_state == CellState::kActive;
  const bool is_high_active = high_state == CellState::kActive;
  const bool is_low_cell = is_low_active || low_state == CellState::kInsideBiggerCell;
  const bool is_high_cell = is_high_active || high_state == CellState::kInsideBiggerCell;
  if ((is_low_active && is_high_cell) || (is_high_active && is_low_cell)) {
    return FacePieceKind::kBetweenCells;
  }
  if (is_high_active && low_state == CellState::kAbsent) {
    return FacePieceKind::kGridEdgeLow;
  }
  if (is_low_active && high_state == CellState::kAbsent) {
    return FacePieceKind::kGridEdgeHigh;
  }
  return FacePieceKind::kNotAPiece;
}

}  // namespace fluidity::grid
