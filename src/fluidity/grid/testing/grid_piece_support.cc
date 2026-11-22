#include "fluidity/grid/testing/grid_piece_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_corners.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {
namespace {

// The block of the piece that holds whole grid block `whole_block`, or kNoIndex when the piece did not load it.
std::uint32_t FindPieceBlock(const PieceTables& piece, std::uint32_t whole_block) {
  const auto found = std::ranges::lower_bound(piece.block_in_whole, whole_block, {}, &CellBlockIndex::value);
  if (found == piece.block_in_whole.end() || found->value != whole_block) {
    return kNoIndex;
  }
  return static_cast<std::uint32_t>(found - piece.block_in_whole.begin());
}

// The whole grid's place for a place of the piece, both of places_per_block places per block; kNoIndex stays.
std::uint32_t ToWholePlace(const PieceTables& piece, std::uint32_t piece_place, std::uint32_t places_per_block) {
  if (piece_place == kNoIndex) {
    return kNoIndex;
  }
  const std::uint32_t piece_block = piece_place / places_per_block;
  return (piece.block_in_whole.at(piece_block).value * places_per_block) + (piece_place % places_per_block);
}

// The piece's place for a place of the whole grid; kNoIndex when its block was not loaded.
std::uint32_t ToPiecePlace(const PieceTables& piece, std::uint32_t whole_place, std::uint32_t places_per_block) {
  if (whole_place == kNoIndex) {
    return kNoIndex;
  }
  const std::uint32_t piece_block = FindPieceBlock(piece, whole_place / places_per_block);
  if (piece_block == kNoIndex) {
    return kNoIndex;
  }
  return (piece_block * places_per_block) + (whole_place % places_per_block);
}

// The two sets of tables of one comparison.
struct TablesPair {
  const GridTables& whole;
  const PieceTables& piece;
  GridTables piece_tables;
};

// The places of a reader's answer, as whole grid indexes: `places` of `count` entries, of the piece when `piece` is
// not null.
template <typename PlaceIndex>
std::vector<std::uint32_t>
ToWholeList(const PlaceIndex* places, int count, const PieceTables* piece, std::uint32_t places_per_block) {
  std::vector<std::uint32_t> whole_places;
  whole_places.reserve(static_cast<std::size_t>(count));
  for (int entry = 0; entry < count; ++entry) {
    const std::uint32_t place = places[entry].value;
    whole_places.push_back(piece == nullptr ? place : ToWholePlace(*piece, place, places_per_block));
  }
  return whole_places;
}

// A difference report: the reader, the kind and whole grid index of the place it started from, and what it was asked.
//
// clang-tidy: three texts, the reader, the kind of place and what was asked; the swappable-parameters check flags them.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::string
ToDifference(std::string_view reader, std::string_view start_kind, std::uint32_t start, std::string_view asked) {
  // NOLINTEND(bugprone-easily-swappable-parameters)
  std::string text(reader);
  text += " from the ";
  text += start_kind;
  text += " at whole place ";
  text += std::to_string(start);
  if (!asked.empty()) {
    text += ", ";
    text += asked;
  }
  return text;
}

// A position of a cell: its low corner plus the fractions of its width on each axis.
Vec3 ComputeCellPoint(const GridTables& tables, CellPlaceIndex cell, std::array<double, 3> width_fractions) {
  const double cell_width = ComputeCellWidth(tables.frame, ToSizeLevel(tables, ToCellBlockIndex(cell)));
  const Vec3 center = ComputeCellCenter(tables, cell);
  return {
      .x = center.x + ((width_fractions.at(0) - 0.5) * cell_width),
      .y = center.y + ((width_fractions.at(1) - 0.5) * cell_width),
      .z = center.z + ((width_fractions.at(2) - 0.5) * cell_width),
  };
}

// Compares FindTouchingCells, FindFacePieces of the cell and of each touching cell, FindCornerPlaces and
// FindCellsAroundEdge from the active cell at whole place `cell`.
std::optional<std::string> FindCellListDifference(const TablesPair& pair, CellPlaceIndex cell) {
  const CellPlaceIndex piece_cell = {.value = ToPiecePlace(pair.piece, cell.value, kCellPlacesPerBlock)};
  const TouchingCells whole_touching = FindTouchingCells(pair.whole, cell);
  const TouchingCells piece_touching = FindTouchingCells(pair.piece_tables, piece_cell);
  if (ToWholeList(whole_touching.cells, whole_touching.count, nullptr, kCellPlacesPerBlock)
      != ToWholeList(piece_touching.cells, piece_touching.count, &pair.piece, kCellPlacesPerBlock)) {
    return ToDifference("FindTouchingCells", "cell", cell.value, "");
  }
  for (int entry = -1; entry < whole_touching.count; ++entry) {
    const CellPlaceIndex read_cell = entry < 0 ? cell : whole_touching.cells[entry];
    const CellFacePieces whole_faces = FindFacePieces(pair.whole, read_cell);
    const CellFacePieces piece_faces =
        FindFacePieces(pair.piece_tables, {.value = ToPiecePlace(pair.piece, read_cell.value, kCellPlacesPerBlock)});
    if (ToWholeList(whole_faces.faces, whole_faces.count, nullptr, kFacePlacesPerBlock)
        != ToWholeList(piece_faces.faces, piece_faces.count, &pair.piece, kFacePlacesPerBlock)) {
      return ToDifference("FindFacePieces", "cell", cell.value, "of cell " + std::to_string(read_cell.value));
    }
  }
  const CellCorners whole_corners = FindCornerPlaces(pair.whole, cell);
  const CellCorners piece_corners = FindCornerPlaces(pair.piece_tables, piece_cell);
  if (ToWholeList(whole_corners.corners, 8, nullptr, kCornerPlacesPerBlock)
      != ToWholeList(piece_corners.corners, 8, &pair.piece, kCornerPlacesPerBlock)) {
    return ToDifference("FindCornerPlaces", "cell", cell.value, "");
  }
  for (const auto& [first_axis, second_axis] :
       {std::array{Axis::kX, Axis::kY}, std::array{Axis::kX, Axis::kZ}, std::array{Axis::kY, Axis::kZ}}) {
    for (int quarter = 0; quarter < 4; ++quarter) {
      const CellsAroundEdge whole_around = FindCellsAroundEdge(pair.whole, cell, first_axis, second_axis, quarter);
      const CellsAroundEdge piece_around =
          FindCellsAroundEdge(pair.piece_tables, piece_cell, first_axis, second_axis, quarter);
      const bool is_same = whole_around.is_same_size == piece_around.is_same_size
          && ToWholeList(whole_around.cells, 4, nullptr, kCellPlacesPerBlock)
              == ToWholeList(piece_around.cells, 4, &pair.piece, kCellPlacesPerBlock);
      if (!is_same) {
        return ToDifference("FindCellsAroundEdge", "cell", cell.value, "quarter " + std::to_string(quarter));
      }
    }
  }
  return std::nullopt;
}

// Compares FindCellAt, with and without a block near it, at the 27 points of the closed box of the active cell at
// whole place `cell`.
std::optional<std::string> FindCellAtDifference(const TablesPair& pair, CellPlaceIndex cell) {
  const CellBlockIndex whole_near = ToCellBlockIndex(cell);
  const CellBlockIndex piece_near = {.value = FindPieceBlock(pair.piece, whole_near.value)};
  for (int point_number = 0; point_number < 27; ++point_number) {
    // Low, middle or high on each axis: 0, 1 or 2 half widths.
    const int x_halves = point_number % 3;
    const int y_halves = (point_number / 3) % 3;
    const int z_halves = point_number / 9;
    const std::array<double, 3> fractions = {0.5 * x_halves, 0.5 * y_halves, 0.5 * z_halves};
    const Vec3 position = ComputeCellPoint(pair.whole, cell, fractions);
    const std::uint32_t whole_found = FindCellAt(pair.whole, position).value;
    const std::uint32_t piece_found =
        ToWholePlace(pair.piece, FindCellAt(pair.piece_tables, position).value, kCellPlacesPerBlock);
    const std::uint32_t whole_found_near = FindCellAt(pair.whole, position, whole_near).value;
    const std::uint32_t piece_found_near =
        ToWholePlace(pair.piece, FindCellAt(pair.piece_tables, position, piece_near).value, kCellPlacesPerBlock);
    if (whole_found != piece_found || whole_found_near != piece_found_near) {
      return ToDifference("FindCellAt", "cell", cell.value, "point " + std::to_string(point_number));
    }
  }
  return std::nullopt;
}

// The face pieces FindNearbyFaces lists for `position` on `axis`, as whole grid indexes, with the error; of the piece
// when `piece` is not null, whose tables are `tables`.
struct NearbyFacesAnswer {
  std::vector<std::uint32_t> faces;
  GridError error = GridError::kNone;
};

NearbyFacesAnswer FindNearbyFacesAnswer(const GridTables& tables, const PieceTables* piece, Vec3 position, Axis axis) {
  NearbyFacesAnswer answer;
  const NearbyFaceList list = FindNearbyFaces(tables, position, axis, CellBlockIndex{kNoIndex}, &answer.error);
  answer.faces = ToWholeList(list.faces, list.count, piece, kFacePlacesPerBlock);
  return answer;
}

// The axis FindNearbyFaces is asked at position `position_number` of the cell at whole place `cell`: the axes take
// turns over the cells and the positions, so every axis is asked, at a third of the cost of all three everywhere.
Axis ToAskedAxis(CellPlaceIndex cell, int position_number) {
  return static_cast<Axis>((cell.value + static_cast<std::uint32_t>(position_number)) % 3);
}

// Compares FindNearbyFaces at 2 positions of the half-open box of the active cell at whole place `cell`, one axis each.
std::optional<std::string> FindCellNearbyFacesDifference(const TablesPair& pair, CellPlaceIndex cell) {
  constexpr std::array<std::array<double, 3>, 2> kPositionFractions = {{{0.0, 0.0, 0.0}, {0.8, 0.3, 0.9}}};
  for (int position_number = 0; position_number < 2; ++position_number) {
    const Vec3 position =
        ComputeCellPoint(pair.whole, cell, kPositionFractions.at(static_cast<std::size_t>(position_number)));
    const Axis axis = ToAskedAxis(cell, position_number);
    const NearbyFacesAnswer whole_answer = FindNearbyFacesAnswer(pair.whole, nullptr, position, axis);
    const NearbyFacesAnswer piece_answer = FindNearbyFacesAnswer(pair.piece_tables, &pair.piece, position, axis);
    if (whole_answer.faces != piece_answer.faces || whole_answer.error != piece_answer.error) {
      return ToDifference("FindNearbyFaces", "cell", cell.value, "position " + std::to_string(position_number));
    }
  }
  return std::nullopt;
}

// Compares FindCellLow, FindCellHigh and FindCornerPlaces from the face piece at whole place `face`, and at
// kFourCellWidths also FindCellsNearFace, the block list of BuildFaceSearch and the walk of FindNextFace.
std::optional<std::string> FindFaceDifference(const TablesPair& pair, FacePlaceIndex face, BorderWidth width) {
  const FacePlaceIndex piece_face = {.value = ToPiecePlace(pair.piece, face.value, kFacePlacesPerBlock)};
  const bool are_cells_same = FindCellLow(pair.whole, face).value
          == ToWholePlace(pair.piece, FindCellLow(pair.piece_tables, piece_face).value, kCellPlacesPerBlock)
      && FindCellHigh(pair.whole, face).value
          == ToWholePlace(pair.piece, FindCellHigh(pair.piece_tables, piece_face).value, kCellPlacesPerBlock);
  if (!are_cells_same) {
    return ToDifference("FindCellLow or FindCellHigh", "face piece", face.value, "");
  }
  const FaceCorners whole_corners = FindCornerPlaces(pair.whole, face);
  const FaceCorners piece_corners = FindCornerPlaces(pair.piece_tables, piece_face);
  if (ToWholeList(whole_corners.corners, 4, nullptr, kCornerPlacesPerBlock)
      != ToWholeList(piece_corners.corners, 4, &pair.piece, kCornerPlacesPerBlock)) {
    return ToDifference("FindCornerPlaces", "face piece", face.value, "");
  }
  if (width != BorderWidth::kFourCellWidths) {
    return std::nullopt;
  }
  const CellsNearFace whole_near = FindCellsNearFace(pair.whole, face);
  const CellsNearFace piece_near = FindCellsNearFace(pair.piece_tables, piece_face);
  if (ToWholeList(whole_near.cells, whole_near.count, nullptr, kCellPlacesPerBlock)
      != ToWholeList(piece_near.cells, piece_near.count, &pair.piece, kCellPlacesPerBlock)) {
    return ToDifference("FindCellsNearFace", "face piece", face.value, "");
  }
  FaceSearch whole_search = BuildFaceSearch(pair.whole, face, kMaxFaceSearchRadius);
  FaceSearch piece_search = BuildFaceSearch(pair.piece_tables, piece_face, kMaxFaceSearchRadius);
  if (ToWholeList(whole_search.blocks, whole_search.block_count, nullptr, 1)
      != ToWholeList(piece_search.blocks, piece_search.block_count, &pair.piece, 1)) {
    return ToDifference("BuildFaceSearch", "face piece", face.value, "");
  }
  for (;;) {
    const std::uint32_t whole_found = FindNextFace(pair.whole, &whole_search).value;
    const std::uint32_t piece_found =
        ToWholePlace(pair.piece, FindNextFace(pair.piece_tables, &piece_search).value, kFacePlacesPerBlock);
    if (whole_found != piece_found) {
      return ToDifference("FindNextFace", "face piece", face.value, "");
    }
    if (whole_found == kNoIndex) {
      return std::nullopt;
    }
  }
}

// Compares, for the active cell at whole place `near_cell`, the face pieces of the set that FindNearbyFaces lists at
// 2 positions of the cell, one axis each: what the transfer step reads to add into a face piece of the set from the
// particles of a cell near it.
std::optional<std::string>
FindTransferDifference(const TablesPair& pair, CellPlaceIndex near_cell, const std::set<std::uint32_t>& set_blocks) {
  const auto keep_set_faces = [&set_blocks](const std::vector<std::uint32_t>& faces) {
    std::vector<std::uint32_t> set_faces;
    for (const std::uint32_t face : faces) {
      if (set_blocks.contains(face / kFacePlacesPerBlock)) {
        set_faces.push_back(face);
      }
    }
    return set_faces;
  };
  if (ToPiecePlace(pair.piece, near_cell.value, kCellPlacesPerBlock) == kNoIndex) {
    return ToDifference("FindNearbyFaces for transfer", "cell near a face piece", near_cell.value, "not loaded");
  }
  constexpr std::array<std::array<double, 3>, 2> kPositionFractions = {{{0.5, 0.5, 0.5}, {0.1, 0.7, 0.4}}};
  for (int position_number = 0; position_number < 2; ++position_number) {
    const Vec3 position =
        ComputeCellPoint(pair.whole, near_cell, kPositionFractions.at(static_cast<std::size_t>(position_number)));
    const Axis axis = ToAskedAxis(near_cell, position_number);
    const NearbyFacesAnswer whole_answer = FindNearbyFacesAnswer(pair.whole, nullptr, position, axis);
    const NearbyFacesAnswer piece_answer = FindNearbyFacesAnswer(pair.piece_tables, &pair.piece, position, axis);
    if (keep_set_faces(whole_answer.faces) != keep_set_faces(piece_answer.faces)) {
      return ToDifference("FindNearbyFaces for transfer", "cell near a face piece", near_cell.value, "");
    }
  }
  return std::nullopt;
}

// Adds to `near_cells` the cells of FindCellsNearFace(face) at most one size level bigger than the face piece.
void ApplyTransferCells(const GridTables& whole, FacePlaceIndex face, std::set<std::uint32_t>& near_cells) {
  const int face_size_level = static_cast<int>(ToSizeLevel(whole, ToCellBlockIndex(face)));
  const CellsNearFace cells_near_face = FindCellsNearFace(whole, face);
  for (int entry = 0; entry < cells_near_face.count; ++entry) {
    const CellPlaceIndex near_cell = cells_near_face.cells[entry];
    if (static_cast<int>(ToSizeLevel(whole, ToCellBlockIndex(near_cell))) >= face_size_level - 1) {
      near_cells.insert(near_cell.value);
    }
  }
}

// Compares every reader from the places of one block of the set, and adds the cells transfer reads to `near_cells`.
std::optional<std::string> FindBlockDifference(
    const TablesPair& pair,
    std::uint32_t block,
    BorderWidth width,
    std::set<std::uint32_t>& near_cells
) {
  for (std::uint32_t place_in_block = 0; place_in_block < kCellPlacesPerBlock; ++place_in_block) {
    const CellPlaceIndex cell = {.value = (block * kCellPlacesPerBlock) + place_in_block};
    if (pair.whole.cell_states[cell.value] != CellState::kActive) {
      continue;
    }
    for (const auto find_difference :
         {&FindCellListDifference, &FindCellAtDifference, &FindCellNearbyFacesDifference}) {
      std::optional<std::string> difference = find_difference(pair, cell);
      if (difference.has_value()) {
        return difference;
      }
    }
  }
  for (std::uint32_t place_in_block = 0; place_in_block < kFacePlacesPerBlock; ++place_in_block) {
    const FacePlaceIndex face = {.value = (block * kFacePlacesPerBlock) + place_in_block};
    if (pair.whole.face_piece_kinds[face.value] == FacePieceKind::kNotAPiece) {
      continue;
    }
    std::optional<std::string> difference = FindFaceDifference(pair, face, width);
    if (difference.has_value()) {
      return difference;
    }
    ApplyTransferCells(pair.whole, face, near_cells);
  }
  for (std::uint32_t place_in_block = 0; place_in_block < kCornerPlacesPerBlock; ++place_in_block) {
    const CornerPlaceIndex corner = {.value = (block * kCornerPlacesPerBlock) + place_in_block};
    const CornerPlaceIndex piece_corner = {.value = ToPiecePlace(pair.piece, corner.value, kCornerPlacesPerBlock)};
    const Vec3 whole_position = ComputeCornerPosition(pair.whole, corner);
    const Vec3 piece_position = ComputeCornerPosition(pair.piece_tables, piece_corner);
    const bool is_same = whole_position.x == piece_position.x && whole_position.y == piece_position.y
        && whole_position.z == piece_position.z;
    if (!is_same) {
      return ToDifference("ComputeCornerPosition", "corner place", corner.value, "");
    }
  }
  return std::nullopt;
}

}  // namespace

GridTables PieceTables::ToTables() const {
  return {
      .frame = frame,
      .version = version,
      .block_count = static_cast<std::uint32_t>(block_ids.size()),
      .block_ids = block_ids.data(),
      .same_size_level_blocks = same_size_level_blocks.data(),
      .bigger_block = bigger_block.data(),
      .smaller_blocks = smaller_blocks.data(),
      .cell_states = cell_states.data(),
      .face_piece_kinds = face_piece_kinds.data(),
  };
}

PieceTables BuildPieceTables(const GridTables& whole, std::span<const CellBlockIndex> loaded_blocks) {
  // Steps: 1 check the list  2 copy block by block  3 mark the face pieces whose other cell was not loaded
  // Step 1.
  PieceTables piece;
  for (std::size_t entry = 0; entry < loaded_blocks.size(); ++entry) {
    const bool is_ascending = entry == 0 || loaded_blocks[entry - 1].value < loaded_blocks[entry].value;
    if (!is_ascending || loaded_blocks[entry].value >= whole.block_count) {
      ADD_FAILURE() << "loaded blocks not ascending or beyond the grid at entry " << entry;
      return piece;
    }
  }
  piece.frame = whole.frame;
  piece.version = whole.version;
  piece.block_in_whole.assign(loaded_blocks.begin(), loaded_blocks.end());
  // Step 2.
  const auto to_piece_entry = [&piece](std::uint32_t whole_block) {
    return whole_block == kNoIndex ? kNoIndex : FindPieceBlock(piece, whole_block);
  };
  for (const CellBlockIndex loaded_block : loaded_blocks) {
    const std::uint64_t block = loaded_block.value;
    piece.block_ids.push_back(whole.block_ids[block]);
    for (std::uint64_t entry = 0; entry < 27; ++entry) {
      piece.same_size_level_blocks.push_back(to_piece_entry(whole.same_size_level_blocks[(block * 27) + entry]));
    }
    piece.bigger_block.push_back(to_piece_entry(whole.bigger_block[block]));
    for (std::uint64_t entry = 0; entry < 8; ++entry) {
      piece.smaller_blocks.push_back(to_piece_entry(whole.smaller_blocks[(block * 8) + entry]));
    }
    for (std::uint64_t entry = 0; entry < kCellPlacesPerBlock; ++entry) {
      piece.cell_states.push_back(whole.cell_states[(block * kCellPlacesPerBlock) + entry]);
    }
    for (std::uint64_t entry = 0; entry < kFacePlacesPerBlock; ++entry) {
      const FacePlaceIndex face = {.value = static_cast<std::uint32_t>((block * kFacePlacesPerBlock) + entry)};
      FacePieceKind kind = whole.face_piece_kinds[face.value];
      // Step 3.
      if (kind == FacePieceKind::kBetweenCells) {
        const std::uint32_t low_block = ToCellBlockIndex(FindCellLow(whole, face)).value;
        const std::uint32_t high_block = ToCellBlockIndex(FindCellHigh(whole, face)).value;
        if (FindPieceBlock(piece, low_block) == kNoIndex || FindPieceBlock(piece, high_block) == kNoIndex) {
          kind = FacePieceKind::kOutsidePiece;
        }
      }
      piece.face_piece_kinds.push_back(kind);
    }
  }
  return piece;
}

std::vector<CellBlockIndex>
FindBorderByReadMarks(const GridTables& whole, std::span<const CellBlockIndex> set, BorderWidth width) {
  std::vector<std::uint8_t> read_marks(whole.block_count, 0);
  std::vector<std::uint8_t> set_marks(whole.block_count, 0);
  for (const CellBlockIndex block : set) {
    set_marks.at(block.value) = 1;
    for (std::uint32_t place_in_block = 0; place_in_block < kCellPlacesPerBlock; ++place_in_block) {
      const CellPlaceIndex cell = {.value = (block.value * kCellPlacesPerBlock) + place_in_block};
      ApplyCellPlaceReadMarks(whole, cell, width, read_marks.data());
    }
    for (std::uint32_t place_in_block = 0; place_in_block < kFacePlacesPerBlock; ++place_in_block) {
      const FacePlaceIndex face = {.value = (block.value * kFacePlacesPerBlock) + place_in_block};
      ApplyFacePlaceReadMarks(whole, face, width, read_marks.data());
    }
  }
  std::vector<CellBlockIndex> border;
  for (std::uint32_t block = 0; block < whole.block_count; ++block) {
    if (read_marks.at(block) != 0 && set_marks.at(block) == 0) {
      border.push_back({.value = block});
    }
  }
  return border;
}

std::optional<std::string> FindFirstReaderDifference(
    const GridTables& whole,
    std::span<const CellBlockIndex> set,
    std::span<const CellBlockIndex> border,
    BorderWidth width
) {
  std::vector<CellBlockIndex> loaded_blocks(set.begin(), set.end());
  loaded_blocks.insert(loaded_blocks.end(), border.begin(), border.end());
  std::ranges::sort(loaded_blocks, {}, &CellBlockIndex::value);
  const PieceTables piece = BuildPieceTables(whole, loaded_blocks);
  const TablesPair pair = {.whole = whole, .piece = piece, .piece_tables = piece.ToTables()};
  std::set<std::uint32_t> set_blocks;
  std::set<std::uint32_t> near_cells;
  for (const CellBlockIndex block : set) {
    set_blocks.insert(block.value);
    std::optional<std::string> difference = FindBlockDifference(pair, block.value, width, near_cells);
    if (difference.has_value()) {
      return difference;
    }
  }
  for (const std::uint32_t near_cell : near_cells) {
    std::optional<std::string> difference = FindTransferDifference(pair, {.value = near_cell}, set_blocks);
    if (difference.has_value()) {
      return difference;
    }
  }
  return std::nullopt;
}

}  // namespace fluidity::grid
