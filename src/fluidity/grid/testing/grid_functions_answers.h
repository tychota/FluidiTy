// The shared grid functions evaluated for one query: the query and result types, and the answer functions, for the GPU
// tests of the grid, private to grid/.
//
// Purpose: give each shared function of the grid tables one query type and one function that answers a query from the
//   tables, so a GPU runner and the CPU loop of a test call the same code.
//
// Idea: the answer functions are marked FLUIDITY_CPU_AND_GPU. A thread of a runner (grid_functions_runner_gpu.h) and
//   the loop of a test on the CPU both call them. The weights are the heavy ones: a thread keeps the list of faces it
//   found and the weights computed from it, 1,152 bytes and more.
//
// Words:
//   query   the input of one thread: a position, a place, or a place and a few numbers
//   walk    the face places a face search gives from its start, in the order FindNextFace gives them
//
// Owns:    FindCellAtQuery, FoundCells, NearbyFaceQuery, NearbyFaceResult, FaceWeightQuery, FaceWeightResult,
//          FormulaQuery, FormulaResult, CellPlaceFacts, FacePlaceFacts, EdgeQuery, FaceSearchQuery, FaceSearchWalk,
//          kMaxFaceSearchWalkFaces, BorderBlockQuery, BorderBlockResult, ComputeBorderBlockResult, MarkedPlaceKind,
//          kMaxMarkedBlocks, ReadMarksQuery, ReadMarksResult, ComputeReadMarksResult,
//          FindCellsAtPosition, ComputeNearbyFaceResult, ComputeFaceWeightResult,
//          ComputeFormulaResult, ComputeCellPlaceFacts, ComputeFacePlaceFacts, FindCellCornersOf, FindFaceCornersOf,
//          ComputeCornerPositionOf, FindCellsAroundEdgeOf, FindTouchingCellsOf, FindFacePiecesOf, FindCellsNearFaceOf,
//          FindFaceSearchWalk
// Uses:    BorderWidth, BlockSet, ComputeIsBorderBlock, ApplyCellPlaceReadMarks, ApplyFacePlaceReadMarks
//          (grid/border_block_steps.h),
//          Axis (base/axis.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), Vec3 (base/vec3.h), CellCorners,
//          FaceCorners, ComputeCornerPosition, FindCellsAroundEdge, CellsAroundEdge (grid/cell_corners.h),
//          ComputeCellWidthsBySizeLevel, ComputeReferenceSpeed (grid/cell_size_rule.h), CellsNearFace,
//          FindCellsNearFace, FaceSearch, BuildFaceSearch, FindNextFace (grid/face_search.h), FindCellAt
//          (grid/find_cell_at.h), GridError (grid/grid_error.h), ComputeSortCount (grid/grid_build_steps.h), the id
//          types and ComputeSmallCell (grid/grid_ids.h), GridTables, ComputeCellCenter, ComputeCellCenterDistance,
//          FindCellPlace, FindFacePlace, ToFacePieceId (grid/grid_tables.h), NearbyFaceList, NearbyFaceMask,
//          NearbyFaceWeights, FaceWeight, FindNearbyFaces, ToMaskOfAll, ComputeNearbyFaceWeights, ComputeFaceWeight,
//          ToNearbyFaceSums (grid/nearby_face_weights.h), TouchingCells, CellFacePieces, FindTouchingCells,
//          FindFacePieces (grid/touching_cells.h), ComputeCompleteWeightedSpread (grid/weight_curve.h)
// Used by: testing/grid_functions_runner_gpu.h, testing/grid_functions_result_words.h and the GPU tests of the grid
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>, <sec-cell-blocks>, <sec-face-pieces>,
//          <sec-nearby-faces>, <sec-face-search>

#pragma once

#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/border_block_steps.h"
#include "fluidity/grid/cell_corners.h"
#include "fluidity/grid/cell_size_rule.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/touching_cells.h"
#include "fluidity/grid/weight_curve.h"

namespace fluidity::grid {

// The query of FindCellAt: a position, and the block a search with a starting block begins from.
struct FindCellAtQuery {
  Vec3 position;  // m, world.
  CellBlockIndex near;  // The starting block of the second form of FindCellAt; kNoIndex when the caller has none.
};

// What both forms of FindCellAt give for one query. 8 bytes and no padding.
struct FoundCells {
  CellPlaceIndex without_start;  // FindCellAt(tables, position).
  CellPlaceIndex with_start;  // FindCellAt(tables, position, near).
};
static_assert(sizeof(FoundCells) == 8, "two 32-bit indexes, with no padding");

// Answers one FindCellAt query with both forms.
FLUIDITY_CPU_AND_GPU inline FoundCells FindCellsAtPosition(const GridTables& tables, const FindCellAtQuery& query) {
  return {
      .without_start = FindCellAt(tables, query.position),
      .with_start = FindCellAt(tables, query.position, query.near),
  };
}

// The query of the nearby faces: a position, an axis, and the faces the caller may keep.
struct NearbyFaceQuery {
  Vec3 position;  // m, world; inside an active cell, or a rounding step from it.
  Axis axis;
  // Bit i of word i / 64 set means the face i of the found list is kept. The answer keeps the faces of the list whose
  // bit is set here, so the mask never holds a bit from list.count on.
  NearbyFaceMask drawn_mask;
};

// What FindNearbyFaces and ComputeNearbyFaceWeights give for one query.
struct NearbyFaceResult {
  NearbyFaceWeights weights;  // All zero when error is not kNone.
  GridError error;  // The error FindNearbyFaces reported.
};

// Answers one query of the nearby faces: finds the list of the position and the axis with no starting block, keeps the
// faces the query's mask allows, and computes the weights. When FindNearbyFaces fails, the weights stay zero.
FLUIDITY_CPU_AND_GPU inline NearbyFaceResult
ComputeNearbyFaceResult(const GridTables& tables, const NearbyFaceQuery& query) {
  NearbyFaceResult result{};
  GridError error = GridError::kInternalFault;
  const NearbyFaceList list = FindNearbyFaces(tables, query.position, query.axis, CellBlockIndex{kNoIndex}, &error);
  result.error = error;
  if (error != GridError::kNone) {
    return result;
  }
  NearbyFaceMask kept = ToMaskOfAll(list);
  for (int word = 0; word < 4; ++word) {
    kept.bits[word] &= query.drawn_mask.bits[word];
  }
  result.weights = ComputeNearbyFaceWeights(tables, list, query.position, kept);
  return result;
}

// The query of ComputeFaceWeight: the weights of a particle, one kept face of its list, and three more faces that show
// the three ways the face form can meet a face outside what the list form holds for that face.
struct FaceWeightQuery {
  NearbyFaceWeights weights;  // From ComputeNearbyFaceWeights on the same tables.
  int position_in_list;  // 0..weights.list.count - 1, and the face is kept.
  FacePlaceIndex
      unlisted_face;  // A face place of the tables on the axis of weights.list; the list may or may not hold it.
  FacePlaceIndex other_axis_face;  // A face place of the tables on another axis than weights.list.
  NearbyFaceSums other_axis_sums;  // ToNearbyFaceSums of the weights of that other axis, at the same position and cell.
};

// What the face form of ComputeFaceWeight and the list form give for one query.
struct FaceWeightResult {
  FaceWeight from_list;  // The list form, for the face at position_in_list.
  FaceWeight seen_from_face;  // The face form, for that same face: the same bits as from_list.
  FaceWeight seen_from_unlisted_face;  // The face form, for unlisted_face, with the sums of weights.
  FaceWeight seen_from_other_axis_face;  // The face form, for other_axis_face, with other_axis_sums.
};

// Answers one ComputeFaceWeight query with the list form and the face form.
FLUIDITY_CPU_AND_GPU inline FaceWeightResult
ComputeFaceWeightResult(const GridTables& tables, const FaceWeightQuery& query) {
  const NearbyFaceWeights& weights = query.weights;
  const FacePlaceIndex listed_face = weights.list.faces[query.position_in_list];
  const NearbyFaceSums sums = ToNearbyFaceSums(weights);
  const CellPlaceIndex cell = weights.list.cell;
  return {
      .from_list = ComputeFaceWeight(tables, weights, query.position_in_list),
      .seen_from_face = ComputeFaceWeight(tables, listed_face, weights.position, cell, sums),
      .seen_from_unlisted_face = ComputeFaceWeight(tables, query.unlisted_face, weights.position, cell, sums),
      .seen_from_other_axis_face =
          ComputeFaceWeight(tables, query.other_axis_face, weights.position, cell, query.other_axis_sums),
  };
}

// Answers one FindCornerPlaces query for a cell place.
FLUIDITY_CPU_AND_GPU inline CellCorners FindCellCornersOf(const GridTables& tables, const CellPlaceIndex& cell) {
  return FindCornerPlaces(tables, cell);
}

// Answers one FindCornerPlaces query for a face place.
FLUIDITY_CPU_AND_GPU inline FaceCorners FindFaceCornersOf(const GridTables& tables, const FacePlaceIndex& face) {
  return FindCornerPlaces(tables, face);
}

// Answers one ComputeCornerPosition query.
FLUIDITY_CPU_AND_GPU inline Vec3 ComputeCornerPositionOf(const GridTables& tables, const CornerPlaceIndex& corner) {
  return ComputeCornerPosition(tables, corner);
}

// The query of FindCellsAroundEdge.
struct EdgeQuery {
  CellPlaceIndex cell;
  Axis first_axis;
  Axis second_axis;
  int quarter;  // 0..3: which of the cell's four edges parallel to the third axis.
};

// Answers one FindCellsAroundEdge query.
FLUIDITY_CPU_AND_GPU inline CellsAroundEdge FindCellsAroundEdgeOf(const GridTables& tables, const EdgeQuery& query) {
  return FindCellsAroundEdge(tables, query.cell, query.first_axis, query.second_axis, query.quarter);
}

// Answers one FindTouchingCells query.
FLUIDITY_CPU_AND_GPU inline TouchingCells FindTouchingCellsOf(const GridTables& tables, const CellPlaceIndex& cell) {
  return FindTouchingCells(tables, cell);
}

// Answers one FindFacePieces query.
FLUIDITY_CPU_AND_GPU inline CellFacePieces FindFacePiecesOf(const GridTables& tables, const CellPlaceIndex& cell) {
  return FindFacePieces(tables, cell);
}

// Answers one FindCellsNearFace query.
FLUIDITY_CPU_AND_GPU inline CellsNearFace FindCellsNearFaceOf(const GridTables& tables, const FacePlaceIndex& face) {
  return FindCellsNearFace(tables, face);
}

// The query of a face search: its start and its radius.
struct FaceSearchQuery {
  FacePlaceIndex face;  // The start.
  double radius_in_cell_widths;  // Widths of a cell of the start's size level, as BuildFaceSearch says.
};

// The most face places a walk keeps.
inline constexpr int kMaxFaceSearchWalkFaces = 1024;

// The walk of one face search. 4,100 bytes and no padding.
struct FaceSearchWalk {
  int faces_found;  // Every face place FindNextFace gave, also those past kMaxFaceSearchWalkFaces.
  // The places in the order FindNextFace gave them: faces[0..min(faces_found, kMaxFaceSearchWalkFaces) - 1].
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  FacePlaceIndex faces[kMaxFaceSearchWalkFaces];
};
static_assert(sizeof(FaceSearchWalk) == 4100, "one 32-bit count and 1,024 32-bit indexes, with no padding");

// Answers one face search query: builds the search, then calls FindNextFace until it gives kNoIndex.
FLUIDITY_CPU_AND_GPU inline FaceSearchWalk FindFaceSearchWalk(const GridTables& tables, const FaceSearchQuery& query) {
  FaceSearchWalk walk{};
  FaceSearch search = BuildFaceSearch(tables, query.face, query.radius_in_cell_widths);
  for (FacePlaceIndex face = FindNextFace(tables, &search); face.value != kNoIndex;
       face = FindNextFace(tables, &search)) {
    if (walk.faces_found < kMaxFaceSearchWalkFaces) {
      walk.faces[walk.faces_found] = face;
    }
    ++walk.faces_found;
  }
  return walk;
}

// The query of the formulas that need no table: one frame, one cell width, the speeds and counts of a build, and a big
// cell.
struct FormulaQuery {
  GridFrame frame{};  // ComputeCellWidthsBySizeLevel; frame.biggest_cell_width is also ComputeReferenceSpeed's width.
  double cell_width = 0;  // m. ComputeCompleteWeightedSpread.
  double largest_liquid_speed = 0;  // m/s. ComputeReferenceSpeed.
  double gravity_size = 0;  // m/s2. ComputeReferenceSpeed.
  std::uint32_t active_count = 0;  // ComputeSortCount.
  std::uint32_t max_cell_blocks = 0;  // ComputeSortCount.
  CellId big_cell{};  // ComputeSmallCell, one size level below kMaxSizeLevels - 1 at most.
  int cell_eighth = 0;  // ComputeSmallCell, 0..7.
};

// What the formulas give for one FormulaQuery.
struct FormulaResult {
  double complete_weighted_spread = 0;  // m2.
  double reference_speed = 0;  // m/s.
  CellWidthsBySizeLevel widths{};  // m.
  std::uint32_t sort_count = 0;
  CellId small_cell{};
};

// Answers one FormulaQuery. The tables are not read: the formulas take none, and the runner passes them as it does for
// every query.
FLUIDITY_CPU_AND_GPU inline FormulaResult
ComputeFormulaResult(const GridTables& /*tables*/, const FormulaQuery& query) {
  return {
      .complete_weighted_spread = ComputeCompleteWeightedSpread(query.cell_width),
      .reference_speed =
          ComputeReferenceSpeed(query.largest_liquid_speed, query.gravity_size, query.frame.biggest_cell_width),
      .widths = ComputeCellWidthsBySizeLevel(query.frame),
      .sort_count = ComputeSortCount(query.active_count, query.max_cell_blocks),
      .small_cell = ComputeSmallCell(query.big_cell, query.cell_eighth),
  };
}

// What the tables say about one active cell place.
struct CellPlaceFacts {
  Vec3 center;  // m, world. ComputeCellCenter.
  CellId id;  // ToCellId of the place.
  CellPlaceIndex place_of_id;  // FindCellPlace of that id: the place again.
};

// Answers what the tables say about one cell place.
FLUIDITY_CPU_AND_GPU inline CellPlaceFacts ComputeCellPlaceFacts(const GridTables& tables, const CellPlaceIndex& cell) {
  const CellId id = ToCellId(tables, cell);
  return {
      .center = ComputeCellCenter(tables, cell),
      .id = id,
      .place_of_id = FindCellPlace(tables, id),
  };
}

// What the tables say about one face place.
struct FacePlaceFacts {
  double center_distance;  // m. ComputeCellCenterDistance.
  FacePieceId id;  // ToFacePieceId of the place, by the tables.
  FacePieceId id_from_parts;  // ToFacePieceId of the axis, size level and high coordinate of the place.
  FacePlaceIndex place_of_id;  // FindFacePlace of that id.
};

// Answers what the tables say about one face place.
FLUIDITY_CPU_AND_GPU inline FacePlaceFacts ComputeFacePlaceFacts(const GridTables& tables, const FacePlaceIndex& face) {
  const CellBlockIndex block = ToCellBlockIndex(face);
  const FacePieceId id = ToFacePieceId(tables, face);
  return {
      .center_distance = ComputeCellCenterDistance(tables, face),
      .id = id,
      .id_from_parts = ToFacePieceId(ToAxis(face), ToSizeLevel(tables, block), ComputeHighPlaceCoord(tables, face)),
      .place_of_id = FindFacePlace(tables, id),
  };
}

// The query of the border block test: one block and one border width, against the block set the runner holds.
struct BorderBlockQuery {
  CellBlockIndex block;  // Below tables.block_count.
  BorderWidth width;
};

// What the border block test says for one BorderBlockQuery: 1 for a border block, 0 for any other. An integer, so the
// result has no padding and compares by its bits.
struct BorderBlockResult {
  std::uint32_t is_border;
};

// Answers one BorderBlockQuery against `set`, whose sorted list must be readable where the caller runs.
FLUIDITY_CPU_AND_GPU inline BorderBlockResult
ComputeBorderBlockResult(const GridTables& tables, const BlockSet& set, const BorderBlockQuery& query) {
  return {.is_border = ComputeIsBorderBlock(tables, set, query.block, query.width) ? 1U : 0U};
}

// Which kind of place a ReadMarksQuery starts from.
enum class MarkedPlaceKind : std::uint8_t {
  kCellPlace,  // ApplyCellPlaceReadMarks.
  kFacePlace,  // ApplyFacePlaceReadMarks.
};

// The most blocks a ReadMarksResult holds a mark for: the room of a random grid (kMaxRandomGridCellBlocks, 252),
// rounded up.
inline constexpr int kMaxMarkedBlocks = 256;

// The query of the forward form of the border: one place, its kind and one border width.
struct ReadMarksQuery {
  std::uint32_t place;  // A cell place or a face place, by `kind`, of block 0..tables.block_count - 1.
  MarkedPlaceKind kind;
  BorderWidth width;
};

// The marks one place makes: marks[b] is 1 for each block its readers read, with its bigger blocks, and 0 for every
// other block. Bytes, so the result has no padding and compares by its bits.
struct ReadMarksResult {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays): a plain array, since GPU code cannot call std::array's member functions.
  std::uint8_t marks[kMaxMarkedBlocks];
};

// Answers one ReadMarksQuery from zeroed marks. Tables of more than kMaxMarkedBlocks blocks give no mark.
FLUIDITY_CPU_AND_GPU inline ReadMarksResult
ComputeReadMarksResult(const GridTables& tables, const ReadMarksQuery& query) {
  ReadMarksResult result{};
  if (tables.block_count > static_cast<std::uint32_t>(kMaxMarkedBlocks)) {
    return result;
  }
  if (query.kind == MarkedPlaceKind::kCellPlace) {
    ApplyCellPlaceReadMarks(tables, CellPlaceIndex{query.place}, query.width, result.marks);
  } else {
    ApplyFacePlaceReadMarks(tables, FacePlaceIndex{query.place}, query.width, result.marks);
  }
  return result;
}

}  // namespace fluidity::grid
