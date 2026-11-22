// The shared grid functions evaluated for one query, and their GPU runners, for the GPU tests of the grid, private to
// grid/.
//
// Purpose: run each shared function of the grid tables on the GPU, one thread per query, so a test can compare every
//   result with the CPU's: exactly for places and counts, within the method paper's tolerance for the weights.
//
// Idea: each shared function gets one query type and one function that answers a query from the tables. The answer
//   functions are marked FLUIDITY_CPU_AND_GPU, so the GPU thread and the CPU loop of the test call the same code. A
//   runner copies the queries to the GPU, runs one thread per query on the tables of a grid built on the GPU, and
//   copies the results back. The weights runner is the heavy one: a thread keeps the list of faces it found and the
//   weights computed from it, 1,152 bytes and more, above the driver's default stack of 1,024 bytes per thread. A
//   test runner reserves nothing: the driver grows the stack at the first launch, as the compiler's frame size for the
//   function says. ComputeWeightsRunnerStackOnGpu reads that frame size, so the hardware run gives each module that
//   calls the weights a measured number for its own stack entry.
//
// Words:
//   query   the input of one thread: a position, a place, or a place and a few numbers
//   walk    the face places a face search gives from its start, in the order FindNextFace gives them
//
// Owns:    FindCellAtQuery, FoundCells, NearbyFaceQuery, NearbyFaceResult, FaceWeightQuery, FaceWeightResult,
//          EdgeQuery, FaceSearchQuery, FaceSearchWalk, kMaxFaceSearchWalkFaces, FindCellsAtPosition,
//          ComputeNearbyFaceResult, ComputeFaceWeightResult, FindCellCornersOf, FindFaceCornersOf,
//          ComputeCornerPositionOf, FindCellsAroundEdgeOf, FindTouchingCellsOf, FindFacePiecesOf, FindCellsNearFaceOf,
//          WalkFaceSearch, FindCellsAtOnGpu, ComputeNearbyFaceResultsOnGpu, ComputeFaceWeightResultsOnGpu,
//          ComputeWeightsRunnerStackOnGpu, FindCellCornersOnGpu, FindFaceCornersOnGpu, ComputeCornerPositionsOnGpu,
//          FindCellsAroundEdgesOnGpu, FindTouchingCellsOnGpu, FindFacePiecesOnGpu, FindCellsNearFacesOnGpu,
//          WalkFaceSearchesOnGpu
// Uses:    Axis (base/axis.h), FLUIDITY_CPU_AND_GPU (base/cpu_and_gpu.h), RunOnCallStatus (base/run_on_call.h), Vec3
//          (base/vec3.h), CellCorners, FaceCorners, ComputeCornerPosition, FindCellsAroundEdge, CellsAroundEdge
//          (grid/cell_corners.h), CellsNearFace, FindCellsNearFace, FaceSearch, BuildFaceSearch, FindNextFace
//          (grid/face_search.h), FindCellAt (grid/find_cell_at.h), GridError (grid/grid_error.h), the id types
//          (grid/grid_ids.h), GridTables (grid/grid_tables.h), NearbyFaceList, NearbyFaceMask, NearbyFaceWeights,
//          FaceWeight, FindNearbyFaces, ToMaskOfAll, ComputeNearbyFaceWeights, ComputeFaceWeight, ToNearbyFaceSums
//          (grid/nearby_face_weights.h), TouchingCells, CellFacePieces, FindTouchingCells, FindFacePieces
//          (grid/touching_cells.h)
// Used by: tests/grid/find_cell_at_on_gpu.test.cc, tests/grid/nearby_face_weights_on_gpu.test.cc and
//          tests/grid/cell_surroundings_on_gpu.test.cc, and testing/grid_functions_runner_gpu.cu, which defines the
//          runners
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>, <sec-cell-blocks>, <sec-face-pieces>,
//          <sec-nearby-faces>, <sec-face-search>

#pragma once

#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/base/cpu_and_gpu.h"
#include "fluidity/base/run_on_call.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/cell_corners.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_error.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/nearby_face_weights.h"
#include "fluidity/grid/touching_cells.h"

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

// The query of ComputeFaceWeight: the weights of a particle, one kept face of its list, and one more face.
struct FaceWeightQuery {
  NearbyFaceWeights weights;  // From ComputeNearbyFaceWeights on the same tables.
  FacePlaceIndex other_face;  // A face place of the tables; it may be outside the list.
  int position_in_list;  // 0..weights.list.count - 1, and the face is kept.
};

// What both forms of ComputeFaceWeight give for one query.
struct FaceWeightResult {
  FaceWeight from_list;  // The first form, for the face at position_in_list.
  FaceWeight seen_from_face;  // The second form, for that same face.
  FaceWeight seen_from_other_face;  // The second form, for other_face.
};

// Answers one ComputeFaceWeight query with both forms.
FLUIDITY_CPU_AND_GPU inline FaceWeightResult
ComputeFaceWeightResult(const GridTables& tables, const FaceWeightQuery& query) {
  const NearbyFaceWeights& weights = query.weights;
  const FacePlaceIndex listed_face = weights.list.faces[query.position_in_list];
  const NearbyFaceSums sums = ToNearbyFaceSums(weights);
  return {
      .from_list = ComputeFaceWeight(tables, weights, query.position_in_list),
      .seen_from_face = ComputeFaceWeight(tables, listed_face, weights.position, weights.list.cell, sums),
      .seen_from_other_face = ComputeFaceWeight(tables, query.other_face, weights.position, weights.list.cell, sums),
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
FLUIDITY_CPU_AND_GPU inline FaceSearchWalk WalkFaceSearch(const GridTables& tables, const FaceSearchQuery& query) {
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

// Each runner below computes the answer function named in its summary for queries[i], for i below `count`, on the GPU,
// into the CPU array `results`. `tables` holds the GPU tables of a built grid (StaggeredGrid::tables(RunOn::kGpu));
// the queries are in CPU memory and are copied to the GPU first. Each writes `results` once the GPU work is done; when
// a later step fails (freeing the GPU memory), the status says so and `results` may already hold the answers.
// Requires: count is 1 or more; queries and results hold count entries; each query meets the preconditions of the
//   shared function it asks.
// Fails: kOutOfMemory  the GPU had no memory for the queries or the results
//        kCallFailed   another GPU call failed

// Runs FindCellsAtPosition over the queries.
RunOnCallStatus
FindCellsAtOnGpu(const GridTables& tables, const FindCellAtQuery* queries, int count, FoundCells* results);

// Runs ComputeNearbyFaceResult over the queries.
RunOnCallStatus ComputeNearbyFaceResultsOnGpu(
    const GridTables& tables,
    const NearbyFaceQuery* queries,
    int count,
    NearbyFaceResult* results
);

// Runs ComputeFaceWeightResult over the queries.
RunOnCallStatus ComputeFaceWeightResultsOnGpu(
    const GridTables& tables,
    const FaceWeightQuery* queries,
    int count,
    FaceWeightResult* results
);

// Runs FindCellCornersOf over the queries.
RunOnCallStatus
FindCellCornersOnGpu(const GridTables& tables, const CellPlaceIndex* queries, int count, CellCorners* results);

// Runs FindFaceCornersOf over the queries.
RunOnCallStatus
FindFaceCornersOnGpu(const GridTables& tables, const FacePlaceIndex* queries, int count, FaceCorners* results);

// Runs ComputeCornerPositionOf over the queries.
RunOnCallStatus
ComputeCornerPositionsOnGpu(const GridTables& tables, const CornerPlaceIndex* queries, int count, Vec3* results);

// Runs FindCellsAroundEdgeOf over the queries.
RunOnCallStatus
FindCellsAroundEdgesOnGpu(const GridTables& tables, const EdgeQuery* queries, int count, CellsAroundEdge* results);

// Runs FindTouchingCellsOf over the queries.
RunOnCallStatus
FindTouchingCellsOnGpu(const GridTables& tables, const CellPlaceIndex* queries, int count, TouchingCells* results);

// Runs FindFacePiecesOf over the queries.
RunOnCallStatus
FindFacePiecesOnGpu(const GridTables& tables, const CellPlaceIndex* queries, int count, CellFacePieces* results);

// Runs FindCellsNearFaceOf over the queries.
RunOnCallStatus
FindCellsNearFacesOnGpu(const GridTables& tables, const FacePlaceIndex* queries, int count, CellsNearFace* results);

// Runs WalkFaceSearch over the queries.
RunOnCallStatus
WalkFaceSearchesOnGpu(const GridTables& tables, const FaceSearchQuery* queries, int count, FaceSearchWalk* results);

// Reads the bytes of local memory per thread that the compiler gave the GPU function of ComputeNearbyFaceResultsOnGpu,
// cudaFuncGetAttributes' localSizeBytes. It is the frame of the function itself and of every function it calls, so a
// module that calls the weights adds its own locals to it for its stack entry.
// Writes `bytes_per_thread` only on success.
// Fails: kCallFailed  the GPU call failed, for example because the code was built for another GPU
RunOnCallStatus ComputeWeightsRunnerStackOnGpu(std::uint64_t* bytes_per_thread);

}  // namespace fluidity::grid
