// The rules that write each entry of a grid piece's piece map and tables, private to grid/.
//
// Purpose: say, for one item at a time, where a loaded block goes in a grid piece, which piece block a neighbour entry
//   names, and what kind a face place has in the piece, so the CPU builder and every GPU thread of a grid piece build
//   write the same bytes.
//
// Idea: a grid piece holds the blocks of a range and of its border, listed in the whole grid's fixed order. Every
//   border block lies before or after the range, so the order of the piece is: the border blocks below the range, the
//   range, the border blocks above it. The loaded items are numbered in a different order, that of the caller's
//   lists: the border entries first, then the range's blocks. ToPiecePlace turns the number of one loaded item into
//   its place in the piece without looking at any other item's place, so the piece map needs no ordering step. A
//   neighbour entry that names a block not loaded reads as none (ToPieceNeighbour). A face piece whose other cell lies
//   in a block not loaded is outside the piece (ComputePieceFaceKind). Example: whole grid of blocks 0..9, range {4, 3}
//   and border {1, 2, 8}. The loaded items 0..5 are the border entries 1, 2, 8 and the range's blocks 4, 5, 6. Their
//   places in the piece are 0, 1, 5, 2, 3, 4, so the piece map is [1, 2, 4, 5, 6, 8]. A neighbour entry naming whole
//   block 7 reads as none; one naming block 5 reads as 3.
//   The build on the GPU runs these rules for one item at a time, one thread each, in five launches that each read
//   only what an earlier launch wrote: the border entries are checked (ComputePieceBorderFailureKey), the piece map is
//   written (ApplyPieceMapWriteAt), each piece block's id and 36 neighbour entries are copied (ApplyPieceBlockCopyAt),
//   each cell place's state is copied (ApplyPieceCellStateCopyAt), and each face place's kind is written
//   (ApplyPieceFaceKindWriteAt). The CPU builder has loops of its own over the same three rules above; a test replays
//   the five functions on the CPU for every thread block and thread and compares with it.
//
// Words:
//   grid piece   one range of a whole grid with its border, loaded for a batch, as a small grid of its own
//   piece map    for each block of a grid piece, its index in the whole grid
//   loaded item  an entry of the border list, or a block of the range: what a grid piece loads from the whole grid
// Owns:    ToPiecePlace, ToPieceNeighbour, ComputePieceFaceKind, PieceLoad, PieceBlocks, PieceTableArrays,
//          kPieceBorderStep, kPieceFaultStep, ComputePieceBorderFailureKey, ComputeIsBorderEntryBelowRange,
//          ApplyPieceMapWriteAt, ApplyPieceBlockCopyAt, ApplyPieceCellStateCopyAt, ApplyPieceFaceKindWriteAt
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), ComputeBorderSoFarFailureKey (grid/border_block_steps.h),
//          ToFailureKey (grid/grid_build_steps.h), GridTables, CellBlockId, CellState, FacePieceKind, FindCellLow,
//          FindCellHigh, ToSizeLevel (grid/grid_tables.h), CellBlockIndex, CellBlockRange, FacePlaceIndex,
//          ToCellBlockIndex, kCellPlacesPerBlock, kFacePlacesPerBlock, kNoIndex (grid/grid_ids.h)
// Used by: grid_piece_cpu.cc, grid_piece_gpu.cu and the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"

namespace fluidity::grid {

// Computes the index, in the grid piece, of loaded item `loaded_item`: items below border_block_count are the border
// entries, the rest are the range's blocks in order.
//
// Method: a border entry i that lies below the range is at i, since every border entry before it lies below the range
//   too; one above the range is at i + range.block_count. The range's block k is at k plus the number of border
//   entries below the range, which a binary search over the ascending border list finds.
// Example: the whole grid of blocks 0..9, range {4, 3}, border {1, 2, 8}: items 0 to 5 give 0, 1, 5, 2, 3, 4.
// Requires: border_blocks holds border_block_count entries, ascending and outside the range; loaded_item is below
//   border_block_count + range.block_count. The border may be null when border_block_count is 0.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr CellBlockIndex ToPiecePlace(
    const CellBlockIndex* border_blocks,
    std::uint32_t border_block_count,
    CellBlockRange range,
    std::uint32_t loaded_item
) noexcept {
  assert(border_blocks != nullptr || border_block_count == 0);
  assert(loaded_item < border_block_count + std::uint64_t{range.block_count});
  if (loaded_item < border_block_count) {
    const bool is_below_range = border_blocks[loaded_item].value < range.first_block;
    return {.value = is_below_range ? loaded_item : loaded_item + range.block_count};
  }
  std::uint32_t low = 0;
  std::uint32_t high = border_block_count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) / 2);
    if (border_blocks[middle].value < range.first_block) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return {.value = low + (loaded_item - border_block_count)};
}

// Computes the index in the grid piece of the whole grid's block `whole_neighbour`: its place in the piece map, or
// kNoIndex when the block is not loaded. kNoIndex stays kNoIndex.
// Method: binary search in the piece map, which is ascending.
// Example: the piece map [1, 2, 4, 5, 6, 8]: whole block 5 gives 3, whole block 7 gives kNoIndex.
// Requires: map holds count entries, ascending with no repeat; it may be null when count is 0.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
ToPieceNeighbour(const CellBlockIndex* map, std::uint32_t count, std::uint32_t whole_neighbour) noexcept {
  assert(map != nullptr || count == 0);
  std::uint32_t low = 0;
  std::uint32_t high = count;
  while (low < high) {
    const std::uint32_t middle = low + ((high - low) / 2);
    if (map[middle].value < whole_neighbour) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  const bool is_found = low < count && map[low].value == whole_neighbour;
  return is_found ? low : kNoIndex;
}

// Computes the kind of one face place of a grid piece's block: the whole grid's kind, except kOutsidePiece for a
// kBetweenCells face piece whose cell_low or cell_high lies in a block not loaded.
//
// Method: the kind is read at the same place in the whole grid's block that the piece block is. A plane-4 place stays
//   kNotAPiece when its high neighbour exists in the whole grid but is not loaded, since the whole grid's kind is
//   kNotAPiece there: no piece is made that the whole grid does not have.
// Example: the piece map [1, 2, 4], and a kBetweenCells place of piece block 0 (whole block 1) whose cell_high is in
//   whole block 3: the kind is kOutsidePiece. With the map [1, 2, 3, 4] it is kBetweenCells.
// Requires: `whole` are the tables of the whole grid; map holds count entries, ascending, each below whole.block_count;
//   piece_block is below count; face_place_in_block is 0..kFacePlacesPerBlock - 1.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr FacePieceKind ComputePieceFaceKind(
    const GridTables& whole,
    const CellBlockIndex* map,
    std::uint32_t count,
    CellBlockIndex piece_block,
    int face_place_in_block
) noexcept {
  assert(piece_block.value < count);
  assert(face_place_in_block >= 0 && face_place_in_block < kFacePlacesPerBlock);
  const FacePlaceIndex face = {
      .value = (map[piece_block.value].value * static_cast<std::uint32_t>(kFacePlacesPerBlock))
          + static_cast<std::uint32_t>(face_place_in_block),
  };
  const FacePieceKind whole_kind = whole.face_piece_kinds[face.value];
  if (whole_kind != FacePieceKind::kBetweenCells) {
    return whole_kind;
  }
  const std::uint32_t low_block = ToCellBlockIndex(FindCellLow(whole, face)).value;
  const std::uint32_t high_block = ToCellBlockIndex(FindCellHigh(whole, face)).value;
  const bool is_low_loaded = ToPieceNeighbour(map, count, low_block) != kNoIndex;
  const bool is_high_loaded = ToPieceNeighbour(map, count, high_block) != kNoIndex;
  return is_low_loaded && is_high_loaded ? FacePieceKind::kBetweenCells : FacePieceKind::kOutsidePiece;
}

// The failure steps of a grid piece's build on the GPU, the high bits of a failure key (ToFailureKey).
inline constexpr int kPieceBorderStep = 1;  // An entry of the border list that is not ascending, outside the range.
inline constexpr int kPieceFaultStep = 2;  // A neighbour entry names a block of the range that the piece map lacks.

// What a grid piece loads: the range and the border list, which the steps read and never write.
struct PieceLoad {
  CellBlockRange range;  // As a run of the whole grid's blocks.
  const CellBlockIndex* border_blocks;  // Ascending, outside the range; null when border_block_count is 0.
  std::uint32_t border_block_count;
};

// The blocks of a grid piece once its piece map is written, which the later steps read.
struct PieceBlocks {
  CellBlockRange range;  // As a run of the whole grid's blocks.
  const CellBlockIndex* map;  // The piece map: the whole grid's index of each piece block, ascending.
  std::uint32_t count;  // Entries of the map: the blocks of the piece.
};

// The arrays of a grid piece that the steps write, each one entry or run per piece block.
struct PieceTableArrays {
  CellBlockId* block_ids;  // 1 per block.
  std::uint32_t* same_size_level_blocks;  // 27 per block.
  std::uint32_t* bigger_block;  // 1 per block.
  std::uint32_t* smaller_blocks;  // 8 per block.
  CellState* cell_states;  // 64 per block.
  FacePieceKind* face_piece_kinds;  // 240 per block.
};

// Computes the failure key of border entry `entry`, or kNoFailureKey: step 2 of the GPU build, one thread for each
// entry, and the check of a replay.
//
// Method: the rule of ComputeBorderSoFarFailureKey, which a border list and the border so far of a search share: an
//   entry fails when it is not above the entry before it, is at or above the whole grid's block count, or is a block of
//   the range. The key is ToFailureKey(kPieceBorderStep, entry, 0), so the smallest key over the entries names the
//   first failing entry whatever the order the threads run in.
// Example: range {4, 3}, 10 blocks, entries {1, 2, 5}: entry 2 fails (5 is a block of the range), the key is
//   ToFailureKey(1, 2, 0).
// Requires: entry is below load.border_block_count; load.border_blocks holds entries up to it.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr std::uint64_t
ComputePieceBorderFailureKey(PieceLoad load, std::uint32_t whole_block_count, std::uint32_t entry) noexcept {
  return ComputeBorderSoFarFailureKey(load.range, load.border_blocks, whole_block_count, entry);
}

// Computes 1 when border entry `entry` lies below the range, else 0. The sum over a valid list is the number of
// border blocks before the range, which is where the range starts in the piece.
// Example: range {4, 3}, entries {1, 2, 8}: entries 0 and 1 give 1, entry 2 gives 0.
// Requires: entry is below load.border_block_count.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t
ComputeIsBorderEntryBelowRange(PieceLoad load, std::uint32_t entry) noexcept {
  assert(entry < load.border_block_count);
  return load.border_blocks[entry].value < load.range.first_block ? 1U : 0U;
}

// Writes the piece map entry of loaded item `loaded_item`: step 3 of the GPU build, one thread for each loaded item.
//
// Method: the entry is at the place ToPiecePlace gives the item, and holds the whole grid's index of the item: the
//   border entry, or the range's first block plus the item's number among the range's blocks. Two items never have
//   one place, so no two threads write one entry, and the step reads only the border list.
// Example: range {4, 3}, border {1, 2, 8}: item 2 (the border entry 8) writes 8 at place 5; item 3 (the range's first
//   block) writes 4 at place 2.
// Requires: the border list passed ComputePieceBorderFailureKey for every entry; loaded_item is below
//   border_block_count + range.block_count; map has that many entries.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr void
ApplyPieceMapWriteAt(PieceLoad load, std::uint32_t loaded_item, CellBlockIndex* map) noexcept {
  assert(map != nullptr);
  const std::uint32_t whole_block = loaded_item < load.border_block_count
      ? load.border_blocks[loaded_item].value
      : load.range.first_block + (loaded_item - load.border_block_count);
  map[ToPiecePlace(load.border_blocks, load.border_block_count, load.range, loaded_item).value] = {
      .value = whole_block,
  };
}

// Copies the block id and the 36 neighbour entries of piece block `piece_block`, translated to the piece: step 4 of
// the GPU build, one thread for each piece block. Gives the failure key of a fault, or kNoFailureKey.
//
// Method: the 27 same-size-level entries, the bigger block and the 8 smaller blocks of the whole grid's block are each
//   turned into a piece index by ToPieceNeighbour, kNoIndex staying kNoIndex. An entry that names a block of the range
//   but is not found means the piece map lacks a block of the range: no input causes it, and the key is
//   ToFailureKey(kPieceFaultStep, piece_block, the entry's number), the entries numbered 0..26, 27 and 28..35. The step
//   reads the piece map and the whole grid's tables, and writes only the arrays of its own piece block.
// Example: the piece map [1, 2, 4, 5, 6, 8], range {4, 3}: an entry naming whole block 5 is written as 3, one naming
//   block 7 as kNoIndex, and one naming block 3 as kNoIndex.
// Requires: blocks.map is written for every piece block (step 3 has finished); piece_block is below blocks.count;
//   `piece` holds room for that block.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr std::uint64_t ApplyPieceBlockCopyAt(
    const GridTables& whole,
    PieceBlocks blocks,
    PieceTableArrays piece,
    std::uint32_t piece_block
) noexcept {
  assert(piece_block < blocks.count);
  const std::uint32_t whole_block = blocks.map[piece_block].value;
  piece.block_ids[piece_block] = whole.block_ids[whole_block];
  std::uint64_t failure_key = kNoFailureKey;
  // Entry numbers: 0..26 the same size level, 27 the bigger block, 28..35 the smaller blocks.
  for (std::uint32_t entry = 0; entry < 36; ++entry) {
    std::uint32_t whole_neighbour = whole.bigger_block[whole_block];
    std::uint32_t* piece_entry = piece.bigger_block + piece_block;
    if (entry < 27) {
      whole_neighbour = whole.same_size_level_blocks[(std::size_t{whole_block} * 27) + entry];
      piece_entry = piece.same_size_level_blocks + (std::size_t{piece_block} * 27) + entry;
    } else if (entry > 27) {
      whole_neighbour = whole.smaller_blocks[(std::size_t{whole_block} * 8) + (entry - 28)];
      piece_entry = piece.smaller_blocks + (std::size_t{piece_block} * 8) + (entry - 28);
    }
    const std::uint32_t piece_neighbour = ToPieceNeighbour(blocks.map, blocks.count, whole_neighbour);
    *piece_entry = piece_neighbour;
    const bool is_in_range = whole_neighbour >= blocks.range.first_block
        && whole_neighbour - blocks.range.first_block < blocks.range.block_count;
    const bool is_range_block_missing = piece_neighbour == kNoIndex && whole_neighbour != kNoIndex && is_in_range;
    if (is_range_block_missing && failure_key == kNoFailureKey) {
      failure_key = ToFailureKey(kPieceFaultStep, piece_block, static_cast<std::uint8_t>(entry));
    }
  }
  return failure_key;
}

// Copies the state of one cell place of the piece and gives the size level of an active cell, else -1: step 5 of the
// GPU build, one thread for each cell place of the piece.
//
// Method: the place is cell place piece_cell_place % 64 of piece block piece_cell_place / 64, which is whole block
//   map[that block]'s same place. The thread writes only the state of its own place, and the caller adds 1 to the
//   count of the returned size level with an integer atomic add, so the counts do not depend on the order of threads.
// Example: the piece map [1, 2, 4], piece_cell_place 64 + 5: the state of cell place 5 of whole block 2 is written to
//   place 69, and the size level of that block is returned when the cell is active.
// Requires: blocks.map is written (step 3 has finished); piece_cell_place is below 64 blocks.count; cell_states holds
//   64 entries for each piece block.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr int ApplyPieceCellStateCopyAt(
    const GridTables& whole,
    PieceBlocks blocks,
    std::uint32_t piece_cell_place,
    CellState* cell_states
) noexcept {
  const std::uint32_t piece_block = piece_cell_place / static_cast<std::uint32_t>(kCellPlacesPerBlock);
  const std::uint32_t place_in_block = piece_cell_place % static_cast<std::uint32_t>(kCellPlacesPerBlock);
  assert(piece_block < blocks.count);
  const std::uint32_t whole_block = blocks.map[piece_block].value;
  const CellState state =
      whole.cell_states[(whole_block * static_cast<std::uint32_t>(kCellPlacesPerBlock)) + place_in_block];
  cell_states[piece_cell_place] = state;
  if (state != CellState::kActive) {
    return -1;
  }
  return static_cast<int>(ToSizeLevel(whole, {.value = whole_block}));
}

// Writes the kind of one face place of the piece and gives 1 when it is a face piece, else 0: step 6 of the GPU build,
// one thread for each face place of the piece.
//
// Method: ComputePieceFaceKind for place piece_face_place % 240 of piece block piece_face_place / 240, written at the
//   piece's own place. The thread reads the whole grid's tables and the piece map, and writes only its own place.
// Example: as ComputePieceFaceKind; a kind other than kNotAPiece gives 1, kOutsidePiece included.
// Requires: blocks.map is written (step 3 has finished); piece_face_place is below 240 blocks.count; face_piece_kinds
//   holds 240 entries for each piece block.
// Read: docs/method-paper/fluidity-method.typ <sec-cell-blocks>
FLUIDITY_CPU_AND_GPU constexpr std::uint32_t ApplyPieceFaceKindWriteAt(
    const GridTables& whole,
    PieceBlocks blocks,
    std::uint32_t piece_face_place,
    FacePieceKind* face_piece_kinds
) noexcept {
  const std::uint32_t piece_block = piece_face_place / static_cast<std::uint32_t>(kFacePlacesPerBlock);
  const int place_in_block = static_cast<int>(piece_face_place % static_cast<std::uint32_t>(kFacePlacesPerBlock));
  const FacePieceKind kind =
      ComputePieceFaceKind(whole, blocks.map, blocks.count, {.value = piece_block}, place_in_block);
  face_piece_kinds[piece_face_place] = kind;
  return kind != FacePieceKind::kNotAPiece ? 1U : 0U;
}

}  // namespace fluidity::grid
