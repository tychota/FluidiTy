// The GPU runners of the GPU tests of the shared grid functions, private to grid/.
//
// Purpose: run each shared function of the grid tables on the GPU, one thread per query, so a test can compare every
//   result with the CPU's: equal bits, for places, counts and doubles alike.
//
// Idea: a runner copies the queries to the GPU, runs the answer function of grid_functions_answers.h with one thread
//   per query on the tables of a grid built on the GPU, and copies the results back. The weights runner is the heavy
//   one: a thread keeps the list of faces it found and the weights computed from it, 1,152 bytes and more, above the
//   driver's default stack of 1,024 bytes per thread. A test runner reserves nothing: the driver grows the stack at
//   the first launch, as the compiler's frame size for the function says. ReadRunnerFunctionInfo reads those frame
//   sizes, and ptxas prints them in the build log (`-Xptxas=-v`).
//   When a runner fails, the status is only kCallFailed or kOutOfMemory; ReadLastRunnerFailure says which CUDA call
//   failed and with which error.
//
// Owns:    FindCellsAtOnGpu, ComputeNearbyFaceResultsOnGpu, ComputeFaceWeightResultsOnGpu, FindCellCornersOnGpu,
//          FindFaceCornersOnGpu, ComputeCornerPositionsOnGpu, FindCellsAroundEdgesOnGpu, FindTouchingCellsOnGpu,
//          FindFacePiecesOnGpu, FindCellsNearFacesOnGpu, WalkFaceSearchesOnGpu, ComputeFormulasOnGpu,
//          ComputeCellPlaceFactsOnGpu, ComputeFacePlaceFactsOnGpu, RunnerFunctionInfo, ReadRunnerFunctionInfo,
//          ReadLastRunnerFailure
// Uses:    RunOnCallStatus (base/run_on_call.h), GridTables (grid/grid_tables.h), the queries and results of
//          grid/testing/grid_functions_answers.h
// Used by: the GPU tests of the grid in tests/grid/ (find_cell_at.gpu.test.cc, nearby_face_weights.gpu.test.cc,
//          cell_surroundings.gpu.test.cc, shared_grid_functions.gpu.test.cc), and testing/grid_functions_runner_gpu.cu,
//          which defines the runners
// Read:    docs/method-paper/fluidity-method.typ <sec-run-on>, <sec-cell-blocks>, <sec-face-pieces>,
//          <sec-nearby-faces>, <sec-face-search>
//
// Shared functions that no answer function above names, each with the public function that reaches it. A test that runs
// the public function runs them too. tools/shared_grid_functions_test.py fails when a function marked
// FLUIDITY_CPU_AND_GPU in a header of src/fluidity/grid is named neither in grid_functions_answers.h nor in this file
// nor in cell_size_rule_runner_gpu.h.
//
// Reached by FindCellsAtPosition, which FindCellAt reaches (find_cell_at.gpu.test.cc):
//   ComesBefore, ComputeBiggerCellCoord, ComputeCellWidth, ComputeIsEarlierInFixedOrder, ComputeLocalPlaceCoord,
//   ComputePackedId, ComputeUnpackedCoord, ComputeUnpackedSizeLevel, FindActiveCellBelow, FindBiggestBlockFrom,
//   FindCellAtCoord, FindCellBlock, FindSortedBlock, ToBiasedSmallestCellCoord, ToBiggestBlockId, ToCellBlockId,
//   ToCellCoord
//
// Reached by ComputeNearbyFaceResult, through FindNearbyFaces and ComputeNearbyFaceWeights
//   (nearby_face_weights.gpu.test.cc):
//   ApplyAscendingInsert, ApplyFacePiecesBesideSplitPlace, ApplyNearbyFaceInsert, ApplyTouchingSmallCells,
//   ComputeCandidateWindowQuarter, ComputeCholeskyInverse, ComputeCholeskySolution, ComputeCoordAlongAxis,
//   ComputeFaceArea, ComputeFaceCenterInQuarterWidths, ComputeFacePosition, ComputeIsActiveCellPlace,
//   ComputeIsCompleteNeighbourhood, ComputeIsInCandidateWindow, ComputeIsKept, ComputeKeptFaceWeight,
//   ComputeMovedAlongAxis, ComputeRawFaceWeight, ComputeScaledFaceWeight, ComputeSideBeyondBlock,
//   ComputeWeightCurve, FindActiveCellHolding, FindHighSideFacePlace, FindLocalCellPlace, FindNearbyFacesWithRoom,
//   FindSmallCellBlock, ToBlockLocalCoord, ToFacePlaceIndex, ToFacePlaceParts
//
// Reached by ComputeFaceWeightResult, through the face form of ComputeFaceWeight (nearby_face_weights.gpu.test.cc):
//   ComputeIsCandidateFace, FindCellHigh, FindCellLow
//
// Reached by FindCellCornersOf, through FindCornerPlaces (cell_surroundings.gpu.test.cc):
//   FindCornerPlaceInSizeLevel, FindCornerPlaceOfPoint
//
// Reached by FindCellsNearFaceOf, through FindCellsNearFace (cell_surroundings.gpu.test.cc):
//   ApplyCellNearFaceInsert
//
// Reached by FindFaceSearchWalk, through BuildFaceSearch and FindNextFace (cell_surroundings.gpu.test.cc):
//   ApplyFaceSearchBlockInsert, ComputeClampedFaceSearchRadius, ComputeIsCoordInRange,
//   ComputeIsWithinQuarterRadius, FindBlockAtCoord
//
// Reached by ComputeFormulaResult, through ComputeCellWidthsBySizeLevel (shared_grid_functions.gpu.test.cc):
//   ComputeTableSizeLevelCount
//
// Reached by ComputeCellPlaceFacts, through ComputeCellCenter and FindCellPlace (shared_grid_functions.gpu.test.cc):
//   ToCellPlaceIndex
//
// Reached by the GPU build of a grid, StaggeredGrid with RunOn::kGpu (build_grid.gpu.test.cc, and every test that
//   builds its grid on the GPU):
//   ApplyAbsentStateAt, ApplyBlockIdWriteAt, ApplyBlockNeighboursWriteAt, ApplyBuildStatusResult,
//   ApplyCoverAnswerAt, ApplyCoverAnswerCopyAt, ApplyFacePieceKindWriteAt, ApplyFailureCells, ApplySortTailFillAt,
//   ApplySortedCellStateAt, ApplySplitCellWritesAt, CheckCellId, CheckOverlapAt, CheckSiblingsAt, CheckSizeStepAt,
//   ComputeBiasedCorner, ComputeBigCell, ComputeBigCellAtSizeLevel, ComputeBlockFaultKey, ComputeBlockIdAt,
//   ComputeBlockIndexOfSortedEntry, ComputeBlockSide, ComputeBlockStartMark, ComputeCellEighth,
//   ComputeCellIdFailureKey, ComputeCellPlacesFailureKey, ComputeCellStateAfterCover, ComputeCellStateBeside,
//   ComputeCoveredCellState, ComputeFacePieceKind, ComputeFailureCells, ComputeHasEarlierStepFailed,
//   ComputeIsCellPresent, ComputeIsCoveredByActiveBigCell, ComputeLocalCoordInNeighbour, ComputeNeighbourEntry,
//   ComputeResetBuildStatus, ComputeRoomFailureKey, ComputeSortKeyOfCorner, ComputeSortedCellFailureKey,
//   ComputeSplitCellCountToAdd, ComputeStoredCellState, FindBlockNeighbours, FindSortedBlockAt, FindSortedCell,
//   FindSortedCellAt, ToBlockSortKey, ToCellSortKey, ToFailureDetail, ToFailureKey, ToFailurePosition,
//   ToFailureStep
//
// Reached by the cell size rule runner, ComputeCellSizeRuleOnGpu and ComputeNewCellSizeLevelsOnGpu
//   (cell_size_rule.gpu.test.cc):
//   ComputeCellSizeFactorCode, ComputeCellSizeFactorTable, ComputeNewCellSizeLevel,
//   ComputeRefinementBoxReachesOutside, ComputeRefinementEntryIndex, ComputeRefinementLayerExtent,
//   ComputeWantedCellWidth, ComputeWantedSizeLevel, FindNearestCommittedCell

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "fluidity/base/run_on_call.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/testing/grid_functions_answers.h"

namespace fluidity::grid {

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

// Runs FindFaceSearchWalk over the queries.
RunOnCallStatus
WalkFaceSearchesOnGpu(const GridTables& tables, const FaceSearchQuery* queries, int count, FaceSearchWalk* results);

// Runs ComputeFormulaResult over the queries.
RunOnCallStatus
ComputeFormulasOnGpu(const GridTables& tables, const FormulaQuery* queries, int count, FormulaResult* results);

// Runs ComputeCellPlaceFacts over the queries.
RunOnCallStatus
ComputeCellPlaceFactsOnGpu(const GridTables& tables, const CellPlaceIndex* queries, int count, CellPlaceFacts* results);

// Runs ComputeFacePlaceFacts over the queries.
RunOnCallStatus
ComputeFacePlaceFactsOnGpu(const GridTables& tables, const FacePlaceIndex* queries, int count, FacePlaceFacts* results);

// What cudaFuncGetAttributes says about one GPU function of the runners.
struct RunnerFunctionInfo {
  std::string name;  // The runner that starts it.
  std::uint64_t local_bytes = 0;  // localSizeBytes: the local memory per thread that the compiler gave the function.
  int register_count = 0;  // numRegs: the registers per thread.
  int most_threads_per_block = 0;  // maxThreadsPerBlock: the largest block this function can start with.
};

// Reads cudaFuncGetAttributes of the GPU function of every runner above, in the order of the runners. Loading the
// attributes loads the code image for the device, so it fails when the binary has none for this GPU. It launches
// nothing, so it says nothing about whether the stack is enough: the runners themselves show that.
// Writes `functions` only on success.
// Fails: kCallFailed  a GPU call failed, for example because the code was built for another GPU
RunOnCallStatus ReadRunnerFunctionInfo(std::vector<RunnerFunctionInfo>* functions);

// The text of the last CUDA call that failed in a runner of this thread, such as "copy of the results to the CPU:
// cudaErrorIllegalAddress (an illegal memory access was encountered)": the step, cudaGetErrorName and
// cudaGetErrorString. Empty after a runner that succeeded. A failed allocation names the error that cudaMalloc left.
// A test prints it after a status that is not kDone, because the status alone does not say which of the five or more
// causes it was.
std::string ReadLastRunnerFailure();

}  // namespace fluidity::grid
