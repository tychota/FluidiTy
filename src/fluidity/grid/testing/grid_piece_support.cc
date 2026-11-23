#include "fluidity/grid/testing/grid_piece_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fluidity/base/axis.h"
#include "fluidity/base/run_on.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_corners.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_piece.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/staggered_grid.h"
#include "fluidity/grid/testing/grid_support.h"
#include "fluidity/grid/testing/staggered_grid_cases.h"
#include "fluidity/grid/touching_cells.h"

namespace fluidity::grid {
namespace {

// The block of the piece that holds whole grid block `whole_block`, or kNoIndex when the piece did not load it.
std::uint32_t FindPieceBlock(const LoadedPieceTables& piece, std::uint32_t whole_block) {
  const auto found = std::ranges::lower_bound(piece.block_in_whole, whole_block, {}, &CellBlockIndex::value);
  if (found == piece.block_in_whole.end() || found->value != whole_block) {
    return kNoIndex;
  }
  return static_cast<std::uint32_t>(found - piece.block_in_whole.begin());
}

// The whole grid's place for a place of the piece, both of places_per_block places per block; kNoIndex stays.
std::uint32_t ToWholePlace(const LoadedPieceTables& piece, std::uint32_t piece_place, std::uint32_t places_per_block) {
  if (piece_place == kNoIndex) {
    return kNoIndex;
  }
  const std::uint32_t piece_block = piece_place / places_per_block;
  return (piece.block_in_whole[piece_block].value * places_per_block) + (piece_place % places_per_block);
}

// The piece's place for a place of the whole grid; kNoIndex when its block was not loaded.
std::uint32_t ToPiecePlace(const LoadedPieceTables& piece, std::uint32_t whole_place, std::uint32_t places_per_block) {
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
  const LoadedPieceTables& piece;
  GridTables piece_tables;
};

// The places of a reader's answer, as whole grid indexes: `places` of `count` entries, of the piece when `piece` is
// not null.
template <typename PlaceIndex>
std::vector<std::uint32_t>
ToWholeList(const PlaceIndex* places, int count, const LoadedPieceTables* piece, std::uint32_t places_per_block) {
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

NearbyFacesAnswer
FindNearbyFacesAnswer(const GridTables& tables, const LoadedPieceTables* piece, Vec3 position, Axis axis) {
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

// The positions of a cell at which transfer is run, as fractions of the cell's width from its low corner, each in the
// middle of its window quarter: quarters (0, 0, 0), (3, 3, 3), and the centre of the cell, the middle of quarter 2 on
// each axis. Every axis meets its lowest quarter 0 and its highest quarter 3, and the centre.
constexpr std::array<std::array<double, 3>, 3> kTransferFractions = {
    std::array{0.125, 0.125, 0.125},
    std::array{0.875, 0.875, 0.875},
    std::array{0.5, 0.5, 0.5},
};

// One particle of a cell that can contribute to a face piece of the set: its position, the faces of the set that
// FindNearbyFaces lists for it on each axis on the whole grid, and its sums on each axis, made on the whole grid as the
// transfer step carries them with the particle.
struct TransferParticle {
  Vec3 position{};
  std::array<std::vector<std::uint32_t>, 3> whole_faces;
  std::array<NearbyFaceSums, 3> sums{};
};

// What the transfer evidence of one comparison keeps: the pair of tables, the blocks of the set, and the particles of
// each cell met so far, by the cell's whole place.
struct TransferEvidence {
  const TablesPair& pair;
  const std::set<std::uint32_t>& set_blocks;
  std::map<std::uint32_t, std::vector<TransferParticle>> particles_of_cell;
};

// Says whether two doubles have the same bits.
bool ComputeIsSameBits(double first, double second) {
  return std::bit_cast<std::uint64_t>(first) == std::bit_cast<std::uint64_t>(second);
}

// Says whether two vectors have the same bits on each axis.
bool ComputeIsSameVectorBits(Vec3 first, Vec3 second) {
  return ComputeIsSameBits(first.x, second.x) && ComputeIsSameBits(first.y, second.y)
      && ComputeIsSameBits(first.z, second.z);
}

// Makes the particles of the active cell at whole place `cell`: one at each of kTransferFractions, with the sums of
// each axis made on the whole grid from a list that keeps every face.
std::vector<TransferParticle> BuildTransferParticles(const GridTables& whole, CellPlaceIndex cell) {
  std::vector<TransferParticle> particles;
  for (const std::array<double, 3>& fractions : kTransferFractions) {
    TransferParticle particle;
    particle.position = ComputeCellPoint(whole, cell, fractions);
    for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
      GridError error = GridError::kNone;
      const NearbyFaceList list = FindNearbyFaces(whole, particle.position, axis, CellBlockIndex{kNoIndex}, &error);
      EXPECT_EQ(error, GridError::kNone);
      const NearbyFaceWeights weights = ComputeNearbyFaceWeights(whole, list, particle.position, ToMaskOfAll(list));
      particle.sums.at(static_cast<std::size_t>(axis)) = ToNearbyFaceSums(weights);
      particle.whole_faces.at(static_cast<std::size_t>(axis)) =
          ToWholeList(list.faces, list.count, nullptr, kFacePlacesPerBlock);
    }
    particles.push_back(particle);
  }
  return particles;
}

// Compares the faces of the set that FindNearbyFaces lists, on each axis, for every particle of the cell at whole place
// `cell`, in the piece and on the whole grid (the whole grid's lists were made with the particles): what the transfer
// step reads to add into a face piece of the set.
std::optional<std::string> FindListedFacesDifference(
    const TablesPair& pair,
    const std::set<std::uint32_t>& set_blocks,
    CellPlaceIndex cell,
    const std::vector<TransferParticle>& particles
) {
  const auto keep_set_faces = [&set_blocks](const std::vector<std::uint32_t>& faces) {
    std::vector<std::uint32_t> set_faces;
    for (const std::uint32_t face : faces) {
      if (set_blocks.contains(face / kFacePlacesPerBlock)) {
        set_faces.push_back(face);
      }
    }
    return set_faces;
  };
  for (const TransferParticle& particle : particles) {
    for (const Axis axis : {Axis::kX, Axis::kY, Axis::kZ}) {
      const NearbyFacesAnswer piece_answer =
          FindNearbyFacesAnswer(pair.piece_tables, &pair.piece, particle.position, axis);
      if (keep_set_faces(particle.whole_faces.at(static_cast<std::size_t>(axis)))
          != keep_set_faces(piece_answer.faces)) {
        return ToDifference("FindNearbyFaces for transfer", "cell near a face piece", cell.value, "");
      }
    }
  }
  return std::nullopt;
}

// Compares what the face piece at whole place `face` does to collect from the particle of cell `cell`, in the piece and
// on the whole grid: ComputeFaceWeight with the particle's sums made on the whole grid. The answers must have the same
// bits.
std::optional<std::string> FindFaceWeightDifference(
    const TablesPair& pair,
    FacePlaceIndex face,
    CellPlaceIndex cell,
    const TransferParticle& particle
) {
  const FacePlaceIndex piece_face = {.value = ToPiecePlace(pair.piece, face.value, kFacePlacesPerBlock)};
  const CellPlaceIndex piece_cell = {.value = ToPiecePlace(pair.piece, cell.value, kCellPlacesPerBlock)};
  const NearbyFaceSums& sums = particle.sums.at(static_cast<std::size_t>(ToAxis(face)));
  const FaceWeight whole_weight = ComputeFaceWeight(pair.whole, face, particle.position, cell, sums);
  const FaceWeight piece_weight = ComputeFaceWeight(pair.piece_tables, piece_face, particle.position, piece_cell, sums);
  const bool is_same = ToWholePlace(pair.piece, piece_weight.face.value, kFacePlacesPerBlock) == whole_weight.face.value
      && ComputeIsSameBits(whole_weight.weight, piece_weight.weight)
      && ComputeIsSameVectorBits(whole_weight.offset, piece_weight.offset)
      && ComputeIsSameVectorBits(whole_weight.offset_from_mean, piece_weight.offset_from_mean)
      && ComputeIsSameVectorBits(whole_weight.weight_change, piece_weight.weight_change);
  if (!is_same) {
    return ToDifference(
        "ComputeFaceWeight",
        "face piece",
        face.value,
        "particle of cell " + std::to_string(cell.value)
    );
  }
  return std::nullopt;
}

// Finds the cells of FindCellsNearFace(face) at most one size level bigger than the face piece, in the tables of
// `tables`, as whole grid indexes: `piece` is null for the whole grid's own tables.
std::vector<std::uint32_t>
FindTransferCells(const GridTables& tables, const LoadedPieceTables* piece, FacePlaceIndex face) {
  const int face_size_level = static_cast<int>(ToSizeLevel(tables, ToCellBlockIndex(face)));
  const CellsNearFace cells_near_face = FindCellsNearFace(tables, face);
  std::vector<std::uint32_t> transfer_cells;
  for (int entry = 0; entry < cells_near_face.count; ++entry) {
    const CellPlaceIndex near_cell = cells_near_face.cells[entry];
    if (static_cast<int>(ToSizeLevel(tables, ToCellBlockIndex(near_cell))) >= face_size_level - 1) {
      transfer_cells.push_back(
          piece == nullptr ? near_cell.value : ToWholePlace(*piece, near_cell.value, kCellPlacesPerBlock)
      );
    }
  }
  return transfer_cells;
}

// Compares what transfer needs for the face piece at whole place `face` of the set: the cells that can contribute to
// it (FindCellsNearFace at most one size level bigger), in the piece and on the whole grid; for each, the faces of the
// set that FindNearbyFaces lists for its particles; and what the face does to collect from each particle,
// ComputeFaceWeight with sums made on the whole grid, bit for bit. A block of a contributing cell that the piece lacks
// shows as a cell missing from the piece's list.
std::optional<std::string> FindTransferDifference(TransferEvidence& evidence, FacePlaceIndex face) {
  const TablesPair& pair = evidence.pair;
  const FacePlaceIndex piece_face = {.value = ToPiecePlace(pair.piece, face.value, kFacePlacesPerBlock)};
  const std::vector<std::uint32_t> whole_cells = FindTransferCells(pair.whole, nullptr, face);
  if (whole_cells != FindTransferCells(pair.piece_tables, &pair.piece, piece_face)) {
    return ToDifference("FindCellsNearFace for transfer", "face piece", face.value, "");
  }
  for (const std::uint32_t whole_cell : whole_cells) {
    const CellPlaceIndex cell = {.value = whole_cell};
    if (!evidence.particles_of_cell.contains(whole_cell)) {
      std::vector<TransferParticle> particles = BuildTransferParticles(pair.whole, cell);
      std::optional<std::string> difference = FindListedFacesDifference(pair, evidence.set_blocks, cell, particles);
      if (difference.has_value()) {
        return difference;
      }
      evidence.particles_of_cell.emplace(whole_cell, std::move(particles));
    }
    for (const TransferParticle& particle : evidence.particles_of_cell.at(whole_cell)) {
      std::optional<std::string> difference = FindFaceWeightDifference(pair, face, cell, particle);
      if (difference.has_value()) {
        return difference;
      }
    }
  }
  return std::nullopt;
}

// Compares every reader from the places of one block of the set, and the transfer evidence of each face piece of it.
std::optional<std::string> FindBlockDifference(TransferEvidence& evidence, std::uint32_t block, BorderWidth width) {
  const TablesPair& pair = evidence.pair;
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
    difference = FindTransferDifference(evidence, face);
    if (difference.has_value()) {
      return difference;
    }
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
  const GridTables no_tables;
  const LoadedPieceTables loaded_tables = {
      .tables = no_tables,
      .block_in_whole = piece.block_in_whole,
  };
  const auto to_piece_entry = [&loaded_tables](std::uint32_t whole_block) {
    return whole_block == kNoIndex ? kNoIndex : FindPieceBlock(loaded_tables, whole_block);
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
        if (FindPieceBlock(loaded_tables, low_block) == kNoIndex
            || FindPieceBlock(loaded_tables, high_block) == kNoIndex) {
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

namespace {

// Says whether `count` items of two arrays have the same bytes.
template <typename Value>
bool ComputeIsSameArray(const Value* first, const std::vector<Value>& second) {
  return second.empty() || std::memcmp(first, second.data(), second.size() * sizeof(Value)) == 0;
}

}  // namespace

std::optional<PieceBuffersRoom> BuildPieceBuffersRoom(std::uint32_t max_cell_blocks) {
  std::expected<TestGrid, GridFailure> owner = BuildTestGrid(
      ToInputs(ToUnitFrame(1), {}),
      {.max_cell_blocks = std::max(max_cell_blocks, 1U), .run_on = RunOn::kCpu}
  );
  EXPECT_TRUE(owner.has_value());
  if (!owner.has_value()) {
    return std::nullopt;
  }
  GridBuffers buffers = TakeEmptyGridBuffers(owner);
  return PieceBuffersRoom{.owner = std::move(*owner), .buffers = std::move(buffers)};
}

std::optional<std::string> FindFirstTableDifference(const GridPiece& piece, const PieceTables& expected) {
  const GridTables tables = piece.grid().tables(RunOn::kCpu);
  if (tables.block_count != expected.block_ids.size()) {
    return "block count";
  }
  const GridFrame frame = tables.frame;
  const bool is_same_frame = frame.origin.x == expected.frame.origin.x && frame.origin.y == expected.frame.origin.y
      && frame.origin.z == expected.frame.origin.z && frame.biggest_cell_width == expected.frame.biggest_cell_width
      && frame.size_level_count == expected.frame.size_level_count;
  if (!is_same_frame) {
    return "frame";
  }
  if (tables.version.value != expected.version.value || piece.grid().version().value != expected.version.value) {
    return "version";
  }
  if (tables.block_count == 0) {
    return std::nullopt;
  }
  const bool is_same_ids = std::ranges::equal(
      std::span(tables.block_ids, tables.block_count),
      expected.block_ids,
      {},
      &CellBlockId::value,
      &CellBlockId::value
  );
  if (!is_same_ids) {
    return "block ids";
  }
  if (!ComputeIsSameArray(tables.same_size_level_blocks, expected.same_size_level_blocks)) {
    return "same size level blocks";
  }
  if (!ComputeIsSameArray(tables.bigger_block, expected.bigger_block)) {
    return "bigger block";
  }
  if (!ComputeIsSameArray(tables.smaller_blocks, expected.smaller_blocks)) {
    return "smaller blocks";
  }
  if (!ComputeIsSameArray(tables.cell_states, expected.cell_states)) {
    return "cell states";
  }
  if (!ComputeIsSameArray(tables.face_piece_kinds, expected.face_piece_kinds)) {
    return "face piece kinds";
  }
  const std::span<const CellBlockIndex> map = piece.block_in_whole_grid();
  if (!std::ranges::equal(map, expected.block_in_whole, {}, &CellBlockIndex::value, &CellBlockIndex::value)) {
    return "piece map";
  }
  const GridCounts& counts = piece.grid().counts();
  const auto face_piece_count =
      static_cast<std::uint64_t>(std::ranges::count_if(expected.face_piece_kinds, [](FacePieceKind kind) {
        return kind != FacePieceKind::kNotAPiece;
      }));
  const auto active_cell_count =
      static_cast<std::uint64_t>(std::ranges::count(expected.cell_states, CellState::kActive));
  const bool is_same_counts = counts.cell_blocks == expected.block_ids.size() && counts.face_pieces == face_piece_count
      && counts.active_cells == active_cell_count;
  return is_same_counts ? std::nullopt : std::optional<std::string>("counts");
}

std::optional<std::string> FindFirstReaderDifference(
    const GridTables& whole,
    const LoadedPieceTables& piece,
    std::span<const CellBlockIndex> set,
    BorderWidth width
) {
  const TablesPair pair = {.whole = whole, .piece = piece, .piece_tables = piece.tables};
  std::set<std::uint32_t> set_blocks;
  for (const CellBlockIndex block : set) {
    set_blocks.insert(block.value);
  }
  TransferEvidence evidence = {.pair = pair, .set_blocks = set_blocks, .particles_of_cell = {}};
  for (const CellBlockIndex block : set) {
    std::optional<std::string> difference = FindBlockDifference(evidence, block.value, width);
    if (difference.has_value()) {
      return difference;
    }
  }
  return std::nullopt;
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
  const LoadedPieceTables loaded_tables = {.tables = piece.ToTables(), .block_in_whole = piece.block_in_whole};
  return FindFirstReaderDifference(whole, loaded_tables, set, width);
}

}  // namespace fluidity::grid
