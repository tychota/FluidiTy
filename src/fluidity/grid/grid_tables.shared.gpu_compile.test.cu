// Proves that the shared headers grid/cell_size_rule.h, grid/grid_build_status.h, grid/grid_build_steps.h,
// grid/grid_ids.h, grid/grid_tables.h, grid/find_cell_at.h, grid/touching_cells.h, grid/cell_corners.h,
// grid/face_search.h and grid/weight_curve.h compile as GPU code:
// label no_gpu.
//
// The GPU function below is never launched. Compiling it in the CUDA container makes the CUDA compiler generate GPU
// code for every function of cell_size_rule.h, grid_build_steps.h, grid_ids.h, the grid table headers and
// weight_curve.h, the helpers those functions call included, and for CellBlockRange, GridFrame, CellState,
// FacePieceKind, GridTables, TouchingCells, CellFacePieces, CellCorners, FaceCorners, CellsAroundEdge, CellsNearFace,
// FaceSearch, WeightCurveValue, GridBuildStatus, CellCheck, CellSizeSettings, CellWidthsBySizeLevel, CellSizeFacts,
// CellSizeFactorTable and RefinementFactors read and written by GPU code. The integer results go to `results`, and the
// double results of weight_curve.h, the grid table headers and cell_size_rule.h go to `double_results`. The CUDA
// compiler generates GPU code for a shared function only where GPU code calls it, so each function is called here once.
// A shared function that calls a CPU-only function fails to compile in the CUDA container. That compile is the proof;
// the test only takes the function's address, which needs no GPU.

#include <gtest/gtest.h>

#include <cstdint>

#include "fluidity/base/axis.h"
#include "fluidity/base/vec3.h"
#include "fluidity/grid/cell_corners.h"
#include "fluidity/grid/cell_size_rule.h"
#include "fluidity/grid/face_search.h"
#include "fluidity/grid/find_cell_at.h"
#include "fluidity/grid/grid_build_status.h"
#include "fluidity/grid/grid_build_steps.h"
#include "fluidity/grid/grid_ids.h"
#include "fluidity/grid/grid_tables.h"
#include "fluidity/grid/touching_cells.h"
#include "fluidity/grid/weight_curve.h"

namespace fluidity::grid {
namespace {

// Reads its inputs from `values` (4 integers), the range, the frame, the two tables and the sorted cells and blocks,
// writes split cells and the id 0 to `cell_ids`, block ids to `block_ids`, the neighbour tables of one block to
// `block_tables` (27 + 1 + 8 entries per sorted block) and states and kinds to `written_cell_states` and
// `written_face_piece_kinds`, and writes at least one integer per shared function to `results` and the doubles of
// weight_curve.h, the grid table headers and cell_size_rule.h to `double_results`, so the compiler cannot drop any
// call.
__global__ void ComputeWithEverySharedGridFunction(
    const std::int32_t* values,
    CellBlockRange range,
    GridFrame frame,
    const CellState* cell_states,
    const FacePieceKind* face_piece_kinds,
    GridBuildStatus* build_status,
    const CellId* sorted_cells,
    const CellBlockId* sorted_blocks,
    std::uint32_t sorted_count,
    CellId* cell_ids,
    CellBlockId* block_ids,
    std::uint32_t* block_tables,
    CellState* written_cell_states,
    FacePieceKind* written_face_piece_kinds,
    std::uint64_t* results,
    double* double_results,
    const std::int8_t* volume_codes
) {
  const auto size_level = static_cast<SizeLevel>(values[0] & 7);
  const CellCoord cell_coord = {.x = values[1], .y = values[2], .z = values[3]};
  const auto axis = static_cast<Axis>(values[0] % 3);

  int result_index = 0;
  results[result_index++] = ComputeIsCoordInRange(cell_coord) ? 1 : 0;
  const CellId cell = ToCellId(size_level, cell_coord);
  results[result_index++] = cell.value;
  results[result_index++] = static_cast<std::uint64_t>(ToSizeLevel(cell));
  results[result_index++] = static_cast<std::uint64_t>(ToCellCoord(cell).z);
  const CellBlockId block = ToCellBlockId(cell);
  results[result_index++] = block.value;
  results[result_index++] = ToFacePieceId(axis, size_level, cell_coord).value;
  const CellId small_cell = ComputeSmallCell(cell, values[1] & 7);
  results[result_index++] = small_cell.value;
  results[result_index++] = ComputeBigCell(small_cell).value;
  results[result_index++] = static_cast<std::uint64_t>(ComputeCellEighth(small_cell));
  results[result_index++] = ComesBefore(cell, small_cell) ? 1 : 0;
  results[result_index++] = ComesBefore(block, ToCellBlockId(small_cell)) ? 1 : 0;
  const auto place = static_cast<std::uint32_t>(values[2]);
  results[result_index++] = ToCellBlockIndex(CellPlaceIndex{place}).value - range.first_block;
  results[result_index++] = ToCellBlockIndex(FacePlaceIndex{place}).value + range.block_count;
  results[result_index++] = ToCellBlockIndex(CornerPlaceIndex{place}).value;
  // grid_tables.h: its types, read by GPU code.
  results[result_index++] = static_cast<std::uint64_t>(frame.size_level_count);
  results[result_index++] = frame.biggest_cell_width > frame.origin.z ? 1 : 0;
  results[result_index++] = cell_states[place] == CellState::kActive ? 1 : 0;
  results[result_index++] = face_piece_kinds[place] == FacePieceKind::kBetweenCells ? 1 : 0;
  // grid_build_status.h: the record, read and written by GPU code.
  build_status->failure_key = values[0] == 0 ? kNoFailureKey : build_status->failure_key;
  build_status->error = GridError::kBadInput;
  build_status->cell = cell;
  build_status->active_cells_per_size_level[values[0] & 15] = build_status->face_pieces + build_status->cell_blocks;
  results[result_index++] = build_status->failure_key + static_cast<std::uint64_t>(build_status->error);
  // grid_build_steps.h: the keys, the checks of one sorted position, the failure key and the split cells.
  const int pass = values[3] % kSortKeyPassCount;
  results[result_index++] = ToCellSortKey(cell, pass) + ToBlockSortKey(cell, pass);
  results[result_index++] = static_cast<std::uint64_t>(CheckCellId(frame, cell));
  results[result_index++] = FindSortedCell(sorted_cells, sorted_count, cell);
  results[result_index++] = FindSortedBlock(sorted_blocks, sorted_count, block).value;
  results[result_index++] = ComputeBigCellAtSizeLevel(small_cell, size_level).value;
  results[result_index++] = FindSortedCellAt(sorted_cells, sorted_count, size_level, cell_coord);
  const auto position = static_cast<std::uint32_t>(values[1]);
  const CellCheck overlap_check = CheckOverlapAt(sorted_cells, sorted_count, position);
  const CellCheck sibling_check = CheckSiblingsAt(sorted_cells, sorted_count, position);
  const CellCheck size_step_check = CheckSizeStepAt(sorted_cells, sorted_count, position);
  results[result_index++] = overlap_check.cell.value + sibling_check.other_cell.value + size_step_check.detail;
  const std::uint64_t failure_key = ToFailureKey(values[0] & 7, position, static_cast<std::uint8_t>(values[2]));
  results[result_index++] = static_cast<std::uint64_t>(ToFailureStep(failure_key)) + ToFailurePosition(failure_key)
      + ToFailureDetail(failure_key);
  // The sorted cells stand in for the input list too: only the call matters here.
  const CellCheck failure_cells =
      ComputeFailureCells(frame, sorted_cells, sorted_count, sorted_cells, sorted_count, failure_key);
  results[result_index++] = static_cast<std::uint64_t>(failure_cells.error) + failure_cells.cell.value;
  // The per-item rules of the check of a list on the GPU (<proc-build-grid-in-parallel> P1 to P6 and P12).
  results[result_index++] = ComputeCellIdFailureKey(frame, sorted_cells, position);
  results[result_index++] = ComputeSortedCellFailureKey(sorted_cells, sorted_count, position, kSizeStepRuleStep);
  results[result_index++] = ComputeHasEarlierStepFailed(failure_key, kSiblingStep) ? 1 : 0;
  results[result_index++] = ComputeResetBuildStatus().failure_key;
  ApplyFailureCells(frame, sorted_cells, sorted_count, sorted_cells, sorted_count, build_status);
  results[result_index++] = static_cast<std::uint64_t>(ComputeSplitCellCountToAdd(cell));
  // The per-item rules of the split cells and the block ids (<proc-build-grid-in-parallel> P7 and P8). block_tables
  // stands in for the sums and the order: only the calls matter here.
  ApplySplitCellWritesAt(cell_ids, block_tables, sorted_count, position);
  results[result_index++] = ComputeRoomFailureKey(sorted_count, position, kCellPlacesStep);
  results[result_index++] = ComputeCellPlacesFailureKey(sorted_count, position, place) + kSortCountFaultKey;
  const std::uint32_t sort_count = ComputeSortCount(sorted_count, place);
  ApplySortTailFillAt(cell_ids, sort_count, position);
  results[result_index++] = ComputeBlockStartMark(cell_ids, block_tables, position);
  results[result_index++] = ComputeBlockIndexOfSortedEntry(cell_ids, block_tables, block_tables, position);
  ApplyBlockIdWriteAt(block_ids, cell_ids, block_tables, block_tables, position);
  // grid_build_steps.h: the table entries of one block, one cell place and one face place, read through GridTables.
  std::uint32_t* same_size_level_blocks = block_tables;
  std::uint32_t* bigger_block = block_tables + (27 * static_cast<std::uint64_t>(sorted_count));
  std::uint32_t* smaller_blocks = bigger_block + sorted_count;
  FindBlockNeighbours(
      sorted_blocks,
      sorted_count,
      CellBlockIndex{place},
      same_size_level_blocks,
      bigger_block,
      smaller_blocks
  );
  const GridTables tables = {
      .frame = frame,
      .version = GridVersion{1},
      .block_count = sorted_count,
      .block_ids = sorted_blocks,
      .same_size_level_blocks = same_size_level_blocks,
      .bigger_block = bigger_block,
      .smaller_blocks = smaller_blocks,
      .cell_states = cell_states,
      .face_piece_kinds = face_piece_kinds,
  };
  // grid_tables.h: the addressing and geometry of the places, read through GridTables.
  const FacePlaceIndex face = {.value = place};
  results[result_index++] = FindCellBlock(tables, block).value + static_cast<std::uint64_t>(ToAxis(face));
  results[result_index++] = static_cast<std::uint64_t>(ToSizeLevel(tables, CellBlockIndex{place}));
  results[result_index++] = ToCellId(tables, CellPlaceIndex{place}).value + ToFacePieceId(tables, face).value;
  results[result_index++] = FindCellLow(tables, face).value + FindCellHigh(tables, face).value;
  results[result_index++] = FindCellPlace(tables, cell).value;
  results[result_index++] = FindFacePlace(tables, ToFacePieceId(axis, size_level, cell_coord)).value;
  results[result_index++] = static_cast<std::uint64_t>(ComputeCoveredCellState(tables, CellPlaceIndex{place}));
  const CellState state_beside = ComputeCellStateBeside(tables, CellBlockIndex{place}, values[1] & 3, 0, 0);
  results[result_index++] = static_cast<std::uint64_t>(state_beside) + tables.version.value;
  results[result_index++] = static_cast<std::uint64_t>(ComputeFacePieceKind(tables, FacePlaceIndex{place}));
  // find_cell_at.h: the active cell that contains a position, by search and from a block near it.
  const Vec3 world_position = {.x = frame.origin.x, .y = frame.origin.y, .z = frame.biggest_cell_width};
  results[result_index++] =
      FindCellAt(tables, world_position).value + FindCellAt(tables, world_position, CellBlockIndex{place}).value;
  // touching_cells.h: the active cells that touch a cell and the face pieces of its sides.
  const TouchingCells touching = FindTouchingCells(tables, CellPlaceIndex{place});
  results[result_index++] = static_cast<std::uint64_t>(touching.count)
      + touching.cells[place % static_cast<std::uint32_t>(kMaxTouchingCells)].value;
  const CellFacePieces cell_face_pieces = FindFacePieces(tables, CellPlaceIndex{place});
  results[result_index++] = static_cast<std::uint64_t>(cell_face_pieces.count)
      + cell_face_pieces.faces[place % static_cast<std::uint32_t>(kMaxCellFacePieces)].value;
  // cell_corners.h: where the corners of a cell and of a face piece are stored, and the cells around an edge.
  const CellCorners cell_corners = FindCornerPlaces(tables, CellPlaceIndex{place});
  const FaceCorners face_corners = FindCornerPlaces(tables, face);
  results[result_index++] = cell_corners.corners[place % 8U].value + face_corners.corners[place % 4U].value;
  const CellsAroundEdge around_edge = FindCellsAroundEdge(tables, CellPlaceIndex{place}, axis, Axis::kZ, values[1] & 3);
  results[result_index++] = around_edge.cells[place % 4U].value + (around_edge.is_same_size ? 1U : 0U);
  // face_search.h: the cells near a face piece and a face search from it.
  const CellsNearFace cells_near_face = FindCellsNearFace(tables, face);
  results[result_index++] = static_cast<std::uint64_t>(cells_near_face.count)
      + cells_near_face.cells[place % static_cast<std::uint32_t>(kMaxCellsNearFace)].value;
  FaceSearch face_search = BuildFaceSearch(tables, face, frame.origin.y);
  results[result_index++] =
      FindNextFace(tables, &face_search).value + static_cast<std::uint64_t>(face_search.block_count);
  // The per-item rules of the tables in parallel (<proc-build-grid-in-parallel> P9 to P12). The written tables stand in
  // for the ones the GridTables reads: only the calls matter here.
  results[result_index++] = ApplyBlockNeighboursWriteAt(
      sorted_blocks,
      sorted_count,
      CellBlockIndex{place},
      same_size_level_blocks,
      bigger_block,
      smaller_blocks
  );
  ApplyAbsentStateAt(written_cell_states, CellPlaceIndex{place});
  const int sorted_size_level =
      ApplySortedCellStateAt(written_cell_states, cell_ids, block_tables, block_tables, sorted_count, position);
  results[result_index++] = static_cast<std::uint64_t>(sorted_size_level + kTableEntryStep);
  GridTables written_tables = tables;
  written_tables.cell_states = written_cell_states;
  written_tables.face_piece_kinds = written_face_piece_kinds;
  results[result_index++] =
      static_cast<std::uint64_t>(ComputeCellStateAfterCover(written_tables, CellPlaceIndex{place}));
  // The end of `results` stands in for the answers of P10 (c): only the calls matter here.
  std::uint64_t* const cover_answers = results + 64;
  ApplyCoverAnswerAt(written_tables, cover_answers, CellPlaceIndex{place});
  ApplyCoverAnswerCopyAt(cover_answers, written_cell_states, CellPlaceIndex{place});
  results[result_index++] = ApplyFacePieceKindWriteAt(written_tables, written_face_piece_kinds, FacePlaceIndex{place});
  ApplyBuildStatusResult(frame, sorted_cells, sorted_count, sorted_cells, sorted_count, build_status);
  results[result_index++] = build_status->active_cells;
  // weight_curve.h: the curve, the spread and the constants.
  // weight_curve.h writes doubles, so they go to their own buffer and no double is converted to an integer.
  int double_result_index = 0;
  const WeightCurveValue curve_value = ComputeWeightCurve(frame.origin.x);
  double_results[double_result_index++] = curve_value.weight;
  double_results[double_result_index++] = curve_value.slope;
  double_results[double_result_index++] = ComputeCompleteWeightedSpread(frame.biggest_cell_width);
  double_results[double_result_index++] = kWeightCurveReach;
  double_results[double_result_index++] = kMinWeightSum;
  double_results[double_result_index++] = kMinPivotShare;
  // grid_tables.h: widths, positions, areas and distances, in m and m2.
  double_results[double_result_index++] = ComputeCellWidth(frame, size_level);
  const Vec3 cell_center = ComputeCellCenter(tables, CellPlaceIndex{place});
  const Vec3 face_position = ComputeFacePosition(tables, face);
  double_results[double_result_index++] = cell_center.x + face_position.y;
  double_results[double_result_index++] = ComputeFaceArea(tables, face) + ComputeCellCenterDistance(tables, face);
  double_results[double_result_index++] = ComputeCornerPosition(tables, CornerPlaceIndex{place}).z;
  // cell_size_rule.h: the widths table, the reference speed, the wanted cell width (m) and the wanted size level.
  const CellWidthsBySizeLevel widths = ComputeCellWidthsBySizeLevel(frame);
  const CellSizeSettings settings = {.smallest_size_level = static_cast<SizeLevel>(frame.size_level_count - 1)};
  const CellSizeFacts facts = {.depth_below_surface = frame.origin.x, .velocity_change_size = frame.origin.y};
  const double reference_speed = ComputeReferenceSpeed(frame.origin.z, 9.81, frame.biggest_cell_width);
  const double wanted_width = ComputeWantedCellWidth(facts, reference_speed, widths, settings);
  double_results[double_result_index++] = wanted_width;
  results[result_index++] = static_cast<std::uint64_t>(ComputeWantedSizeLevel(wanted_width, widths));
  // cell_size_rule.h: the table of cell size factors, and the factor code of a cell in a refinement volume of one
  // entry, whose layer-0 array is `volume_codes`.
  const CellSizeFactorTable factor_table = ComputeCellSizeFactorTable();
  RefinementFactors volume{.count_x = 1, .count_y = 1, .count_z = 1, .entry_spacing_log2 = 0, .layer_count = 1};
  volume.layer_codes[0] = volume_codes;
  const int factor_code = ComputeCellSizeFactorCode(volume, settings.smallest_size_level, sorted_cells[0]);
  double_results[double_result_index++] = factor_table.factor[factor_code + kMaxCellSizeFactorCode];
  // cell_size_rule.h: the nearest committed cell and the size level of a new cell, read through GridTables.
  results[result_index++] = FindNearestCommittedCell(tables, world_position).value;
  results[result_index++] =
      static_cast<std::uint64_t>(ComputeNewCellSizeLevel(tables, world_position, settings.smallest_size_level));
  results[result_index++] = static_cast<std::uint64_t>(kMaxNearbyFaces);
}

TEST(SharedGridFunctions, BuiltForTheGpu_HaveAnAddress) {
  const auto function_address = &ComputeWithEverySharedGridFunction;
  EXPECT_NE(function_address, nullptr);
}

}  // namespace
}  // namespace fluidity::grid
