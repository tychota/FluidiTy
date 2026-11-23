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
//
// Words:
//   grid piece   one range of a whole grid with its border, loaded for a batch, as a small grid of its own
//   piece map    for each block of a grid piece, its index in the whole grid
//   loaded item  an entry of the border list, or a block of the range: what a grid piece loads from the whole grid
// Owns:    ToPiecePlace, ToPieceNeighbour, ComputePieceFaceKind
// Uses:    FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), GridTables, FacePieceKind, FindCellLow, FindCellHigh
//          (grid/grid_tables.h), CellBlockIndex, CellBlockRange, FacePlaceIndex, ToCellBlockIndex, kFacePlacesPerBlock,
//          kNoIndex (grid/grid_ids.h)
// Used by: grid_piece_cpu.cc and the tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-cell-blocks>

#pragma once

#include <cassert>
#include <cstdint>

#include "fluidity/base/cpu_and_gpu.h"
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

}  // namespace fluidity::grid
